

from datetime import datetime
import math
import pytz
import time

import glm
import numpy as np
from termcolor import cprint, colored
import pybullet as p
import pybullet_data

import sensor
import Bullet_util as utl

# import motion_atlas_transformer as motions
import motion_controller as cnt

work_dir = '/home/ubuntu/Desktop/workspace/'
nao_path = 'bullet3/data/humanoid/nao.urdf'
atlas_path = 'pybullet_robots/data/atlas_add_footsensor/atlas_v4_with_multisense.urdf'
ROTATE = 2* np.pi


# 物理シミュレーションのGUIモードで接続
physicsClient = p.connect(p.GUI)  # または p.DIRECT で非表示モード

# 環境初期化
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8/3)
# p.setGravity(0, 0,0)
# リアルタイムで動かすならこれを有効に
p.setRealTimeSimulation(0)  # 0なら手動stepSimulation()

timestep = 0.01
p.setTimeStep(timestep)

# カメラの初期位置
# Yaw:Z軸方向に右ねじの回転系
# Pitch：90(真下)～-90(真上)
p.resetDebugVisualizerCamera(cameraDistance=2,
                             cameraYaw=45,
                             cameraPitch=0,
                             cameraTargetPosition=[0.5,0,0.8])

#モデル読み込み
floor = p.loadURDF('plane.urdf')

atlas1 = p.loadURDF(work_dir + atlas_path, [0, 0, 1.2])

#直立
motion1 = [[0,0,1], [0,0,1], [0,0,1], [0,0,1]]
#足首を伸ばす、ももを曲げる
motion2 = [[1,0,1], [0,0,1], [-1,0,0], [0,0,1]]
cnt.init_pose(atlas1)



rec =None
isRec = False
rec_fps = 20
counter=-50
while True:
    if counter == 100:
        cnt.set_pose(atlas1, motion1)
    if counter == 300:
        cnt.set_pose(atlas1, motion2)
    if counter == 500:
        cnt.set_pose(atlas1, motion1)
    
    if counter == 700:
        counter = 0
        p.resetBasePositionAndOrientation(
           bodyUniqueId=atlas1,              # オブジェクトのID
            posObj=[0, 0, 1.2],                    # 新しい位置
            ornObj=p.getQuaternionFromEuler([0, 0, 0])  # 新しい向き（例：初期化）
        )

    if counter % 50 == 0:
        current_motion = cnt.get_pose(atlas1)
        cnt.motion_viewer(current_motion, glm.vec3(0, 1, 0.5))
        
    counter += 1
    if counter % 50 == 0:
        print(f'frame:{counter}')
        if isRec:print('rec')
    

    # キー入力を取得
    keys = p.getKeyboardEvents()
    # 終了キーを検出
    if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
        print("Pressed Q, exiting.")
        break
    if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
        p.stepSimulation()
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        if not isRec:
            now_str = datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y%m%d_%H%M%S")
            rec = utl.recode('motion_check'+now_str, 960, 540, rec_fps=20)
        isRec = True
        if isRec:cprint('start recode', color='green')
    if ord('e') in keys and keys[ord('e')] & p.KEY_WAS_TRIGGERED:
        isRec = False
        if not isRec:cprint('stop recode', color='green')

    p.stepSimulation()
    if isRec and (counter % (int(1/timestep) / rec_fps) == 0):
        rec.capture()
    time.sleep(timestep)

p.disconnect()