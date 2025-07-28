import copy
import time
from datetime import datetime
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

from motion_controller import set_pose, get_pose, init_pose

atlas_path = 'pybullet_robots/data/atlas_add_footsensor/atlas_v4_with_multisense.urdf'

class AtlasEnv(gym.Env):
    def __init__(self, is_direct=True, render_mode=None):
        super().__init__()
        #クラス特有の初期化
        self.render_mode = render_mode
        self.atlas_init_pos = [0, 0, 1.1]
        
        # PyBulletセットアップ
        if is_direct:
            self.physicsClient = p.connect(p.DIRECT)
        else:
            self.physicsClient = p.connect(p.GUI)

        p.setGravity(0, 0, -9.8/10)
        p.setTimeStep(1.0 / 100)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.floor_id = p.loadURDF("plane.urdf")
        self.atlas_id = p.loadURDF(atlas_path, self.atlas_init_pos)
        init_pose(self.atlas_id)
    

        # 観測空間（例：過去3フレームの姿勢＋次の予定姿勢）
        self.history_length = 3
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
                             2:glm.vec3(0, 0, 1), 3:glm.vec3(0, 0, 1)},
                           {0:glm.vec3(0, 0, 1), 1:glm.vec3(0, 0, 1),
                             2:glm.vec3(0, 0, 1), 3:glm.vec3(0, 0, 1)},
                           {0:glm.vec3(0, 0, 1), 1:glm.vec3(0, 0, 1),
                             2:glm.vec3(0, 0, 1), 3:glm.vec3(0, 0, 1)},
                           ]  # 過去3ステップ分

        #目標の姿勢        
        self.target_pose = []
        for i in range(1000):
            self.target_pose.append({0:glm.vec3(0, 0, 1), 1:glm.vec3(0, 0, 1),
                                    2:glm.vec3(0, 0, 1), 3:glm.vec3(0, 0, 1)})
        
    def reset(self, seed=None, options=None):
        # p.resetSimulation()
        # self.floor_id = p.loadURDF("plane.urdf")
        # self.atlas_id = p.loadURDF(atlas_path, [0, 0, 1.2])
        self.__reset(self.atlas_init_pos)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0 / 100)
        init_pose(self.atlas_id)
        self.current_step = 0

        self.past_poses = [{0:glm.vec3(0, 0, 1), 1:glm.vec3(0, 0, 1),
                             2:glm.vec3(0, 0, 1), 3:glm.vec3(0, 0, 1)},
                           {0:glm.vec3(0, 0, 1), 1:glm.vec3(0, 0, 1),
                             2:glm.vec3(0, 0, 1), 3:glm.vec3(0, 0, 1)},
                           {0:glm.vec3(0, 0, 1), 1:glm.vec3(0, 0, 1),
                             2:glm.vec3(0, 0, 1), 3:glm.vec3(0, 0, 1)},
                           ]  # 過去3ステップ分
        next_pose = self.target_pose[0]
        obs = self.__create_next_observationSpace(self.past_poses, next_pose)

        info = {'is_reset':True}

        return obs, info


    def step(self, action):
        #Actionを変形する
        #もしもActionが0ベクトルを返した場合はそこは前のモーションを使用する
        motion = self.__action2motion(action)

        #モーションをシミュレーターに適応
        set_pose(self.atlas_id, motion)
        p.stepSimulation()
        #シミュレーション結果を取得
        current_pose = get_pose(self.atlas_id)
        #現在の姿勢の更新
        self.past_poses.pop(0)
        self.past_poses.append(current_pose)

        #本来の目標の姿勢を取得
        now_frame_target_motion = self.target_pose[self.current_step]

        #報酬計算
        reward = self.__reward(current_pose, now_frame_target_motion)
        #姿勢を長く維持できるほど報酬
        reward += 0.5
        
        #次のステップに移行
        self.current_step += 1


        #終了要件を満たしたなら終了
        #転倒などを評価
        terminated = self.__terminate_episode(current_pose)

        #時間切れで強制狩猟        
        truncated = (self.current_step >= self.max_steps)
        if truncated:
            next_pose = self.target_pose[-1]
        else:
            next_pose = self.target_pose[self.current_step]

        #次のイベント用の入力を作成
        obs = self.__create_next_observationSpace(self.past_poses, next_pose)
        
        #debug用のInfo
        info = {}
        info['is_reset'] = True
        info['reward'] = reward

        return obs, reward, terminated, truncated, info


    def render(self):
        # cprint('render run', 'red')
        if self.render_mode=='human':
            return True
        elif self.render_mode=='rgb_array':
            width, height = 960, 720
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[10, 10, -15],      # カメラの位置
                cameraTargetPosition=[0, 0, 0],  # 注視点
                cameraUpVector=[0, 0, 1]                       # 上方向（Z軸が上）
            )
            
            (_, _, px, _, _) = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix
            # projectionMatrix=proj_matrix
            ,renderer=p.ER_TINY_RENDERER
            )
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
            vec = vec / norm
            ret[i] = glm.vec3(*vec)
        return ret

    def __create_next_observationSpace(self, pose_histry, next_pose):
        #それぞれのモーションデータは数値インデックスのglm.vec3形式
        #それを1タイムステップごとに1つとし、List形式で保存
        #それらを1次元のnumpy配列に変換する

        ret = []

        #次のモーションを取得
        # next_motion = next_pose
        # current_pose_and_next_motion = copy.deepcopy(pose_histry)
        # current_pose_and_next_motion.append(next_motion)

        #これまでの姿勢たちをリストに変形
        #1つのモーションにつき0~3でインデクスされた辞書であり
        #1つの要素につきglm.vec3であるのでそれぞれをListに変形
        for motion in pose_histry: #current_pose_and_next_motion:
            #過去3ステップの姿勢について1ステップごとに処理
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
        reward_weights = [0.5, 0.7, 0.7, 1.2]
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
        head_height = curentpose[1] * .5 + curentpose[2] * .5 + curentpose[3] *.8
        if head_height.z < 0.7:
            return True
        return False
    
    def __reset(self, initial_pos):
        p.resetBasePositionAndOrientation(
           bodyUniqueId=self.atlas_id,              # オブジェクトのID
            posObj=initial_pos,                    # 新しい位置
            ornObj=p.getQuaternionFromEuler([0, 0, 0])  # 新しい向き（例：初期化）
        )
        num_joints = p.getNumJoints(self.atlas_id)
        
        # 各ジョイントをゼロにリセット
        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(self.atlas_id, joint_index)
            joint_type = joint_info[2]

            # 可動ジョイントのみ（REVOLUTE or PRISMATIC）を対象にリセット
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                p.resetJointState(self.atlas_id, joint_index, targetValue=0.0)



##########################################
#独自処理のためのBasecallback継承クラス
##########################################
class CustomCallback(BaseCallback):
    def __init__(self, render_interval=20, save_dir='renders', verbose=0):
        '''
        render_intervalのエピソード毎に保存する
        '''
        super().__init__(self)
        self.render_interval = render_interval

        self.n_episode = 0
        self.is_first_of_episode = False
        cv2.namedWindow('rendering', cv2.WINDOW_NORMAL)

    def _on_step(self):
        # self.training_env[0].render()
        # return super()._on_step()




        dones = self.locals['dones']
        if any(dones):
            #1つのエピソードが終わるたびに呼ばれる処理
            print(f'next episode is {self.n_episode}')
            self.n_episode += 1
            self.is_first_of_episode = True
            
            if (self.n_episode % self.render_interval) == 0:
                #指定ステップごとに現在の様子を表示する
                # client_id = p.connect(p.GUI)
                env = AtlasEnv(is_direct=False, render_mode='rgb_array')
                obs, _ = env.reset()
                done = False
                total_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    #動画として保存も検討
                    img = env.render()
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imshow('rendering', img)
                    cv2.waitKey(1)
                    done = terminated or truncated
                    total_reward += reward
                print(f'This episode is :{self.n_episode}, reward is {total_reward}')
                env.close()
                # p.disconnect(client_id)



        else:
            self.is_first_of_episode = False


        return True
    
    def _on_rollout_start(self):
        return True

    def _on_training_start(self):
        cprint('on training stard called', 'red')
    


#####################
#実行環境
#####################

total_timesteps = 1_000_000
    
env = AtlasEnv(is_direct=True, render_mode='rgb_array')
    # check_env(env)



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




from stable_baselines3.common.logger import configure


custom_calback = CustomCallback(render_interval=100)

calbacks = CallbackList([
    eval_callback,
    custom_calback
])

model = PPO('MlpPolicy', env, verbose=1)

new_logger = configure(log_path, ["stdout", "csv"])  # CSV出力！
model.set_logger(new_logger)

model.learn(total_timesteps=total_timesteps, log_interval=10, 
            callback=calbacks, progress_bar=True)



# model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
model.save('atlas_rl_model_standing')
# obs, _ = env.reset()
# for _ in range(1000):
#     action, _ = model.predict()
#     obs, reward, done, truncated, info = env.step(action)
