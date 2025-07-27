import copy
import time

import numpy as np
import glm

import pybullet as p
import pybullet_data

import gymnasium as gym
from gymnasium import spaces
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from motion_controller import set_pose, get_pose, init_pose

atlas_path = 'pybullet_robots/data/atlas_add_footsensor/atlas_v4_with_multisense.urdf'

class AtlasEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # PyBulletセットアップ
        
        self.physicsClient = p.connect(p.GUI)
        p.setGravity(0, 0, -9.8/10)
        p.setTimeStep(1.0 / 100)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.floor_id = p.loadURDF("plane.urdf")
        self.atlas_id = p.loadURDF(atlas_path, [0, 0, 1.2])
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
        
        self.target_pose = []
        for i in range(1000):
            self.target_pose.append({0:glm.vec3(0, 0, 1), 1:glm.vec3(0, 0, 1),
                                    2:glm.vec3(0, 0, 1), 3:glm.vec3(0, 0, 1)})
        
    def reset(self, seed=None, options=None):

        p.resetSimulation()
        self.floor_id = p.loadURDF("plane.urdf")
        self.atlas_id = p.loadURDF(atlas_path, [0, 0, 1.2])
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


        return obs, {}

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
        
        #次のステップに移行
        self.current_step += 1


        #終了要件を満たしたなら終了
        done = (self.current_step >= self.max_steps)
        if done:
            next_pose = self.target_pose[-1]
        else:
            next_pose = self.target_pose[self.current_step]

        #次のイベント用の入力を作成
        obs = self.__create_next_observationSpace(self.past_poses, next_pose)
        
        #debug用のInfo
        info = {}

        return obs, reward, done, False, info


    def render(self):
        pass  # GUI版なら `p.connect(p.GUI)` にして、ここでstep以外に可視化処理入れてもOK

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
        next_motion = next_pose
        current_pose_and_next_motion = copy.deepcopy(pose_histry)
        current_pose_and_next_motion.append(next_motion)

        #これまでの姿勢たちと次のモーションをリストに変形
        #1つのモーションにつき0~3でインデクスされた辞書であり
        #1つの要素につきglm.vec3であるので
        for motion in current_pose_and_next_motion:
            #過去3ステップの姿勢について1ステップごとに処理
            for vec in motion.values():
                vec = glm.normalize(vec)
                ret.append(list(vec))

        return np.array(ret).flatten().astype(np.float32)
    
    def __reward(self, currentpose, targetpose):
        ret = 0
        for i in range(len(currentpose)):
            cVec = glm.normalize(currentpose[i])
            tVec = glm.normalize(targetpose[i])
            d = glm.clamp(glm.dot(cVec, tVec), -1, 1) #類似度の計算
            ret += d
        return ret / len(currentpose) #平均をとり[-1, 1]に収める

    
env = AtlasEnv()
# check_env(env)
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save('atlas_rl_model_standing')
# obs, _ = env.reset()
# for _ in range(1000):
#     action, _ = model.predict()
#     obs, reward, done, truncated, info = env.step(action)