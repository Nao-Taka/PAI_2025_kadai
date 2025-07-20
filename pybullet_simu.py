import time

import numpy as np
import pybullet as p
import pybullet_data

import sensor

work_dir = '/home/ubuntu/Desktop/workspace/'
nao_path = 'bullet3/data/humanoid/nao.urdf'
ROTATE = 2* np.pi


# 物理シミュレーションのGUIモードで接続
physicsClient = p.connect(p.GUI)  # または p.DIRECT で非表示モード

# 環境初期化
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
timestep = 0.01
p.setTimeStep(timestep)

#モデル読み込み
floor = p.loadURDF('plane.urdf')
startPos = [0, 0, 0.5]
nao = p.loadURDF(work_dir + nao_path, startPos)


# car = p.loadURDF('racecar/racecar.urdf', [0, 0, 0.1])

# for i in range(p.getNumJoints(car)):
#     p.getJointInfo(car, i)
# p.setJointMotorControl2(car, 4, controlMode=p.POSITION_CONTROL, targetPosition=ROTATE/8)
# p.setJointMotorControl2(car, 6, controlMode=p.POSITION_CONTROL, targetPosition=ROTATE/8)
# p.setJointMotorControl2(car, 2, controlMode=p.VELOCITY_CONTROL, targetVelocity=40)
# p.setJointMotorControl2(car, 3, controlMode=p.VELOCITY_CONTROL, targetVelocity=40)
# p.setJointMotorControl2(car, 5, controlMode=p.VELOCITY_CONTROL, targetVelocity=40)
# p.setJointMotorControl2(car, 7, controlMode=p.VELOCITY_CONTROL, targetVelocity=40)
counter=0
while True:
    p.stepSimulation()
    # pos, rot = p.getBasePositionAndOrientation(car)
    # p.resetDebugVisualizerCamera(cameraDistance=3,
    #                              cameraYaw=-90,
    #                              cameraPitch=-70,
    #                              cameraTargetPosition=pos)
    # print('tete')
    if counter %10 == 0:
        sensor.sensor_debug(nao, floor)
    # print(f'position:{pos}, rotate:{rot}')
    time.sleep(timestep*5)
    counter += 1
    if counter % 100==0:
        counter = 0

p.disconnect()