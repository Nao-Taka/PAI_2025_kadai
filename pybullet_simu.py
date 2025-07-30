import time

import numpy as np
from termcolor import cprint, colored
import pybullet as p
import pybullet_data

import sensor
import Bullet_util as utl

work_dir = '/home/ubuntu/Desktop/workspace/'
nao_path = 'bullet3/data/humanoid/nao.urdf'
atlas_path = 'pybullet_robots/data/atlas_add_footsensor/atlas_v4_with_multisense.urdf'
ROTATE = 2* np.pi


# 物理シミュレーションのGUIモードで接続
physicsClient = p.connect(p.GUI)  # または p.DIRECT で非表示モード

# 環境初期化
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8/10)
# p.setGravity(0, 0,0)
# リアルタイムで動かすならこれを有効に
p.setRealTimeSimulation(0)  # 0なら手動stepSimulation()

timestep = 0.01
p.setTimeStep(timestep)

# カメラの初期位置
# Yaw:Z軸方向に右ねじの回転系
# Pitch：90(真下)～-90(真上)
p.resetDebugVisualizerCamera(cameraDistance=1.8,
                             cameraYaw=0,
                             cameraPitch=0,
                             cameraTargetPosition=[0.5,0,0.5])

#モデル読み込み
floor = p.loadURDF('plane.urdf')
startPos = [0, 0, 1]
# nao = p.loadURDF(work_dir + nao_path, startPos)
atlas = p.loadURDF(work_dir + atlas_path, startPos)
# p.getBodyInfo(atlas)


robot_id = atlas
num_joints = p.getNumJoints(robot_id)
# print(f"リンク数: {num_joints}")

for link_index in range(num_joints):
    info = p.getJointInfo(robot_id, link_index)
    link_state = p.getLinkState(robot_id, link_index, computeForwardKinematics=True)
    pos = link_state[0]
    orn = link_state[1]
    link_name = info[12].decode("utf-8")  # 名前（bytes → str）
    joint_name = info[1].decode("utf-8")
    joint_type = info[2]
    limit_lower = info[8]
    limit_upper = info[9]
    # print(f"[{link_index}] Link: {link_name}, Joint: {joint_name}, {pos}/{orn}")
    # print(f"{link_index} {link_name} {pos[0]} {pos[1]} {pos[2]}")
    print(f'name {joint_name} type {joint_type} limit {limit_lower} to {limit_upper}')
    p.addUserDebugPoints([pos], [[0,0,1]], pointSize=10)

# a = 0/0

#モーションのテスト
import bvh2absmot
import motion_atlas_transformer

motiondata = bvh2absmot.bvh2motion('motions/02_01.bvh')
# m_dict = motiondata.get_quaternion(0)
# motion_atlas_transformer.set_motion(atlas, m_dict)
# m_dict = motiondata.get_quaternion(1)
# motion_atlas_transformer.set_motion(atlas, m_dict)
# m_dict = motiondata.get_quaternion(2)
# motion_atlas_transformer.set_motion(atlas, m_dict)
# m_dict = motiondata.get_quaternion(3)
# motion_atlas_transformer.set_motion(atlas, m_dict)
# m_dict = motiondata.get_quaternion(4)
# motion_atlas_transformer.set_motion(atlas, m_dict)
# p.setJointMotorControl2(bodyUniqueId=atlas, jointIndex=29,
#                             controlMode=p.POSITION_CONTROL, targetPosition=ROTATE/4)


# rec = utl.recode('motion_check', 1920, 1080, fps=1/timestep)
isRec = False
counter=-50
while True:
    # p.setJointMotorControl2(bodyUniqueId=atlas, jointIndex=28,
    #                         controlMode=p.POSITION_CONTROL, targetPosition=counter/50)
    # m_dict = motiondata.get_quaternion(counter/5)
    # motion_atlas_transformer.set_motion(atlas, m_dict)
    
#####    p.stepSimulation()
    # rec.capture()
    # pos, rot = p.getBasePositionAndOrientation(car)
    # print('tete')
    # if counter %10 == 0:
    #     sensor.sensor_debug(atlas, floor)
    # print(f'position:{pos}, rotate:{rot}')


    counter += 1
    if counter % 50 == 0:
        motion_atlas_transformer.quaternion_viewer(m_dict)
        print(f'frame:{counter}')

    # キー入力を取得
    keys = p.getKeyboardEvents()
    # 終了キーを検出
    if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
        print("Pressed Q, exiting.")
        break
    if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
        p.stepSimulation()
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        isRec = True
        if isRec:cprint('start recode', color='green')
    if ord('e') in keys and keys[ord('e')] & p.KEY_WAS_TRIGGERED:
        isRec = False
        if not isRec:cprint('stop recode', color='green')
    p.stepSimulation()
    time.sleep(timestep)

p.disconnect()