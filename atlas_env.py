'''
Atlasのモーションの強化学習に必要な環境構成
およびコールバック関数を定義
'''

from datetime import datetime
import time
import os
import pytz

import numpy as np
import glm
from termcolor import cprint, colored

import cv2
import pybullet as p
import pybullet_data

import gymnasium as gym
from gymnasium import spaces
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.env_checker import check_env

from Bullet_util import saveMovie
from motion_controller import set_pose, get_pose, init_pose, init_pose_reset
import make_motion

atlas_path = 'pybullet_robots/data/atlas_add_footsensor/atlas_v4_with_multisense.urdf'

'''
is_direct=Trueで環境を起動
直立以外のポーズを行う場合はset_poseでセットする
学習を開始
'''
class AtlasEnv(gym.Env):
    def __init__(self, is_direct=True, render_mode=None, history_length=3):
        super().__init__()
        #クラス特有の初期化
        self.render_mode = render_mode
        self.atlas_init_pos = [0, 0, 1.]
        
        # PyBulletセットアップ
        if is_direct:
            self.physicsClient = p.connect(p.DIRECT)
        else:
            self.physicsClient = p.connect(p.GUI)

        p.setGravity(0, 0, -9.8, physicsClientId=self.physicsClient)
        p.setTimeStep(1.0 / 100, physicsClientId=self.physicsClient)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physicsClient)
        self.floor_id = p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)
        self.atlas_id = p.loadURDF(atlas_path, self.atlas_init_pos, physicsClientId=self.physicsClient)
        init_pose_reset(self.atlas_id, self.physicsClient)
    

        # 観測空間（例：過去3フレームの姿勢＋次の予定姿勢）
        self.history_length = history_length
        self.vec_dim = 4 * 3  # foot, sune, momo, body の各ベクトル（3次元）×4
        obs_dim = (self.history_length + 1) * self.vec_dim
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    
        # アクション空間（次の姿勢ベクトル1セットを出力）
        act_dim = self.vec_dim  # foot〜body のベクトル（3D × 4）
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

        # ステップカウンタ
        self.current_step = 0
        self.max_steps = 1000

        # 姿勢履歴・目標姿勢の保存
        self.past_poses = [{0:glm.vec3(0, 0, 1), 1:glm.vec3(0, 0, 1),
                             2:glm.vec3(0, 0, 1), 3:glm.vec3(0, 0, 1)}
                             for _ in range(self.history_length)]  # 過去nステップ分

        #目標の姿勢        
        self.target_pose = []
        for i in range(1000):
            self.target_pose.append({0:glm.vec3(0, 0, 1), 1:glm.vec3(0, 0, 1),
                                    2:glm.vec3(0, 0, 1), 3:glm.vec3(0, 0, 1)})
            
    def set_pose(self, motions):
        '''
        [{0:glm.vec3(0, 0, 1), 1:glm.vec3(0, 0, 1),
        2:glm.vec3(0, 0, 1), 3:glm.vec3(0, 0, 1)},
        ]
        の形式でリストを渡す
        '''
        self.target_pose = motions
        self.max_step = len(motions)
    
    def reset(self, seed=None, options=None):
        # p.resetSimulation()
        # self.floor_id = p.loadURDF("plane.urdf")
        # self.atlas_id = p.loadURDF(atlas_path, [0, 0, 1.2])
        self.__reset(self.atlas_init_pos)
        p.setGravity(0, 0, -9.8, physicsClientId=self.physicsClient)
        p.setTimeStep(1.0 / 100, physicsClientId=self.physicsClient)
        init_pose_reset(self.atlas_id, self.physicsClient)
        self.current_step = 0

        self.past_poses = [{0:glm.vec3(0, 0, 1), 1:glm.vec3(0, 0, 1),
                             2:glm.vec3(0, 0, 1), 3:glm.vec3(0, 0, 1)}
                             for _ in range(self.history_length)]  # 過去nステップ分
        next_pose = self.target_pose[0]
        obs = self.__create_next_observationSpace(self.past_poses, next_pose)

        info = {'is_reset':True}

        return obs, info


    def step(self, action):
        info = {}

        #Actionを変形する
        #もしもActionが0ベクトルを返した場合はそこは前のモーションを使用する
        motion = self.__action2motion(action)

        #モーションをシミュレーターに適応
        set_pose(self.atlas_id, motion, self.physicsClient)
        p.stepSimulation(physicsClientId=self.physicsClient)
        #シミュレーション結果を取得
        current_pose = get_pose(self.atlas_id, self.physicsClient)
        #現在の姿勢の更新
        self.past_poses.pop(0)
        self.past_poses.append(current_pose)

        #本来の目標の姿勢を取得
        now_frame_target_motion = self.target_pose[self.current_step]

        #報酬計算
        reward = self.__reward(current_pose, now_frame_target_motion)
        info['pose_reward'] = reward
        
        #姿勢を長く維持できるほど報酬
        reward += 0.5
        
        #次のステップに移行
        self.current_step += 1


        #終了要件を満たしたなら終了
        #転倒などを評価
        terminated = self.__terminate_episode(current_pose)

        #時間切れで強制終了
        truncated = (self.current_step >= self.max_steps)
        if truncated:
            next_pose = self.target_pose[-1]
        else:
            next_pose = self.target_pose[self.current_step]

        #次のイベント用の入力を作成
        obs = self.__create_next_observationSpace(self.past_poses, next_pose)
        
        #debug用のInfo
        info['is_reset'] = True
        info['reward'] = reward

        return obs, reward, terminated, truncated, info


    def render(self):
        # cprint('render run', 'red')
        if self.render_mode=='human':
            return True
        elif self.render_mode=='rgb_array':
            width, height = 960, 720
            p.resetDebugVisualizerCamera(cameraDistance=2,
                                         cameraYaw=45,
                                         cameraPitch=-30,
                                         cameraTargetPosition=self.atlas_init_pos
                                         ,physicsClientId=self.physicsClient)
            width, height, px, _, _ = p.getCameraImage(width=width,
                                                        height=height
                                                        ,physicsClientId=self.physicsClient)
            
            
            rgb_array = np.array(px)[:, :, :3]  # RGBA → RGB
            return rgb_array
        return None
    
    def close(self):
        if p.isConnected(self.physicsClient):
            p.disconnect(self.physicsClient)

    def __action2motion(self, action)-> dict[int, glm.vec3]:
        #actionは12次元のnumpy配列
        ret = {}
        EPS = 1e-6

        vectors = action.reshape(4, 3)
        for i, vec in enumerate(vectors):
            #それぞれの姿勢ベクトルごとに処理
            #0ベクトルなら前の姿勢を設定
            norm = np.linalg.norm(vec)
            if norm < EPS:
                vec = np.array(self.past_poses[-1][i])
            else:
                vec = vec / norm
            ret[i] = glm.vec3(*vec)
        return ret

    def __create_next_observationSpace(self, pose_histry, next_pose):
        #それぞれのモーションデータは数値インデックスのglm.vec3形式
        #それを1タイムステップごとに1つとし、List形式で保存
        #それらを1次元のnumpy配列に変換する

        ret = []

        #これまでの姿勢たちをリストに変形
        #1つのモーションにつき0~3でインデクスされた辞書であり
        #1つの要素につきglm.vec3であるのでそれぞれをListに変形
        for motion in pose_histry:
            #過去nステップの姿勢について1ステップごとに処理
            for vec in motion.values():
                vec = glm.normalize(vec)
                ret.append(list(vec))

        #次のモーションの番
        for vec in next_pose.values():
            vec = glm.normalize(vec)
            ret.append(list(vec))

        return np.array(ret).flatten().astype(np.float32)
    
    def __reward(self, currentpose, targetpose):
        ret = 0
        reward_weights = [1.5, 0.5, 0.5, 1.2]
        for i in range(len(currentpose)):
            cVec = glm.normalize(currentpose[i])
            tVec = glm.normalize(targetpose[i])
            d = glm.clamp(glm.dot(cVec, tVec), -1, 1) #類似度の計算
            ret += d * reward_weights[i]
        return ret / len(currentpose) #平均をとる

    def __terminate_episode(self, curentpose):
        #早期終了のためのメソッド
        #今回の場合は一定のたおれたとき
        #条件を満たした場合はTrue
        hip_height  = curentpose[1] * .5 + curentpose[2] * .5 
        head_height = hip_height + curentpose[3] *.8
        if hip_height.z < 0.5:
            return True
        if head_height.z < 0.7:
            return True
        return False
    
    def __reset(self, initial_pos):
        p.resetBasePositionAndOrientation(
           bodyUniqueId=self.atlas_id,              # オブジェクトのID
            posObj=initial_pos,                    # 新しい位置
            ornObj=p.getQuaternionFromEuler([0, 0, 0])  # 新しい向き（例：初期化）
            , physicsClientId=self.physicsClient
        )
        num_joints = p.getNumJoints(self.atlas_id, physicsClientId=self.physicsClient)
        
        # 各ジョイントをゼロにリセット
        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(self.atlas_id, joint_index, 
                                        physicsClientId=self.physicsClient)
            joint_type = joint_info[2]

            # 可動ジョイントのみ（REVOLUTE or PRISMATIC）を対象にリセット
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                p.resetJointState(self.atlas_id, joint_index, targetValue=0.0,
                                  physicsClientId=self.physicsClient)



##########################################
#独自処理のためのBasecallback継承クラス
##########################################
class CustomCallback(BaseCallback):
    def __init__(self, save_dirpath, render_interval=100, model_save_interval=100,verbose=0):
        '''
        render_intervalのエピソード毎に保存する
        '''
        super().__init__(self)
        self.save_dirpath = save_dirpath
        self.render_interval = render_interval
        self.model_save_interval = model_save_interval
        #保存用フォルダの作成
        os.makedirs(f'{self.save_dirpath}movie/', exist_ok=True)
        os.makedirs(f'{self.save_dirpath}models/', exist_ok=True)

        self.n_episode = 0
        self.is_first_of_episode = False
        self.howlong_score = 0
        cv2.namedWindow('rendering', cv2.WINDOW_NORMAL)

    def _on_step(self):
        #何ステップ続いたかをスコアリング
        self.howlong_score += 1

        dones = self.locals['dones']
        if any(dones):
            #1つのエピソードが終わるたびに呼ばれる処理
            #エピソードの持続時間
            print(f'{self.n_episode}: duration time...: {self.howlong_score}')
            with open(self.save_dirpath + 'duration_log.csv', 'a') as f:
                f.write(f'{self.n_episode}, {self.howlong_score}\n')

            self.howlong_score = 0
            self.n_episode += 1
            self.is_first_of_episode = True
            
            if (self.n_episode % self.render_interval) == 0:
                #指定ステップごとに現在の様子を表示する
                # client_id = p.connect(p.GUI)
                #検証用の環境構築
                env = AtlasEnv(is_direct=False, render_mode='rgb_array')
                obs, _ = env.reset()
                done = False
                total_reward = 0

                rec = saveMovie(f'{self.save_dirpath}movie/{self.n_episode}')
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    #動画として保存
                    # self.training_env[0].render()
                    img = env.render()
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imshow('rendering', img)
                    rec.capture(img)
                    cv2.waitKey(1)
                    done = terminated or truncated
                    total_reward += reward
                print(f'This episode is :{self.n_episode}, reward is {total_reward}')
                env.close()
            
            if (self.n_episode % self.model_save_interval) == 0:
                #定期的なモデルの保存
                self.model.save(f'{self.save_dirpath}models/{self.n_episode}')

        else:
            self.is_first_of_episode = False

        return True
    
    def _on_rollout_start(self):
        return True

    def _on_training_start(self):
        cprint('on training stard called', 'red')
    

if __name__=='__main__':
    #####################
    #実行環境
    #####################
    total_timesteps = 1_000_000
    env = AtlasEnv(is_direct=True, render_mode='rgb_array')
    # check_env(env)
    env.set_pose(make_motion.ozigi())

    # ログ出力用ディレクトリの設定
    now_str = datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y%m%d_%H%M%S")
    log_path = f"./logs/run1_{now_str}/"

    #####################
    #コールバック関係の処理
    #####################

    # 環境の評価とログ出力を行うコールバック
    eval_callback = EvalCallback(env,
                             best_model_save_path=log_path,
                             log_path=log_path, eval_freq=10000,
                             deterministic=True, render=False)

    custom_calback = CustomCallback(save_dirpath=log_path, render_interval=100)

    calbacks = CallbackList([eval_callback,
                        custom_calback])

    model = PPO('MlpPolicy', env, verbose=1)
    model.load('best_model_standing')

    from stable_baselines3.common.logger import configure
    new_logger = configure(log_path, ["stdout", "csv"])  # CSV出力！
    model.set_logger(new_logger)

    model.learn(total_timesteps=total_timesteps, log_interval=10, 
                callback=calbacks, progress_bar=True)

    model.save('atlas_rl_model_standing')
