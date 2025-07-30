#お辞儀やしゃがみなどの簡単なモーションを作成する
import math
import time

import glm
import numpy as np

import pybullet as p
import pybullet_data

from ax5_motion_controller import init_pose_reset, set_pose, get_pose, motion_viewer
from bvh2absmot import bvh2motion


vecName2idx = {'body':0, 
               'Lmomo':1, 'Lsune':2, 'Lfoot':3,
               'Rmomo':4, 'Rsune':5, 'Rfoot':6}

def achiles():
    pose = {0:glm.vec3(0, 0, 1)
                ,1:glm.vec3(1, 0.1, -1), 2:glm.vec3(0, 0, -1), 3:glm.vec3(0, 0, -1)
                ,4:glm.vec3(-1, -0.1, -1), 5:glm.vec3(-2, 0, -1), 6:glm.vec3(0, 0, -1)
                 }
    for i in range(7):
        pose[i] = glm.normalize(pose[i])
    return pose

def standing():
    basepose = {0:glm.vec3(0, 0, 1)
                ,1:glm.vec3(0, 0, -1), 2:glm.vec3(0, 0, -1), 3:glm.vec3(0, 0, -1)
                ,4:glm.vec3(0, 0, -1), 5:glm.vec3(0, 0, -1), 6:glm.vec3(0, 0, -1)
                 }
    ret = []
    for i in range(1000):
        ret.append(basepose)
    return ret


def ozigi():
    motions = []
    #1秒直立
    #3秒お辞儀
    #1秒待機
    #3秒で頭を上げる
    #2秒待機
    basepose = {0:glm.vec3(0, 0, 1)
                ,1:glm.vec3(0, 0, -1), 2:glm.vec3(0, 0, -1), 3:glm.vec3(0, 0, -1)
                ,4:glm.vec3(0, 0, -1), 5:glm.vec3(0, 0, -1), 6:glm.vec3(0, 0, -1)
                 }

    #1秒直立
    for i in range(100):
        motions.append(basepose)

    #3秒でお辞儀
    for i in range(300):
        theta = (math.pi / 3) /300 * i

        motion = {0:glm.vec3(math.sin(theta), 0, math.cos(theta))
                ,1:glm.vec3(0, 0, -1), 2:glm.vec3(0, 0, -1), 3:glm.vec3(0, 0, -1)
                ,4:glm.vec3(0, 0, -1), 5:glm.vec3(0, 0, -1), 6:glm.vec3(0, 0, -1)}
        motions.append(motion)
        
    #1秒キープ
    for i in range(100):
        theta = (math.pi / 3)
        motion = {0:glm.vec3(math.sin(theta), 0, math.cos(theta))
                ,1:glm.vec3(0, 0, -1), 2:glm.vec3(0, 0, -1), 3:glm.vec3(0, 0, -1)
                ,4:glm.vec3(0, 0, -1), 5:glm.vec3(0, 0, -1), 6:glm.vec3(0, 0, -1)}
        motions.append(motion)

    #3秒で頭を上げる
    for i in range(300):
        theta = (math.pi / 3) - (math.pi / 3) /300 * i
        motion = {0:glm.vec3(math.sin(theta), 0, math.cos(theta))
                ,1:glm.vec3(0, 0, -1), 2:glm.vec3(0, 0, -1), 3:glm.vec3(0, 0, -1)
                ,4:glm.vec3(0, 0, -1), 5:glm.vec3(0, 0, -1), 6:glm.vec3(0, 0, -1)}
        motions.append(motion)

    #1秒直立
    for i in range(100):
        motions.append(basepose)

    return motions


def wark():
    walk_bvh = bvh2motion('motions/02_01.bvh')
    walk_quat = []

    walk_quat.extend(walk_bvh.get_quaternions(0, 10))
    walk_quat.extend(walk_bvh.get_quaternions(11, 150))
    walk_quat.extend(walk_bvh.get_quaternions(11, 150))
    walk_quat.extend(walk_bvh.get_quaternions(11, 150))
    walk_quat.extend(walk_bvh.get_quaternions(11, 150))
    walk_quat.extend(walk_bvh.get_quaternions(11, 150))
    walk_quat.extend(walk_bvh.get_quaternions(11, 150))
    walk_quat.extend(walk_bvh.get_quaternions(11, 150))

    qn = bvh2motion.quat_name

    ret = []
    for i in range(len(walk_quat)):
        quat = walk_quat[i]
        basepose = {0:glm.vec3(0, 0, 1)
                ,1:glm.vec3(0, 0, -1), 2:glm.vec3(0, 0, -1), 3:glm.vec3(0, 0, -1)
                ,4:glm.vec3(0, 0, -1), 5:glm.vec3(0, 0, -1), 6:glm.vec3(0, 0, -1)
                 }

        parts = ['body', 'lHip', 'lKnee', 'lAnkle', 'rHip', 'rKnee', 'rAnkle']

        motion = {}
        for i, part in enumerate(parts):
            q = quat[qn[part]]
            if part in ['lAnkle', 'rAnkle']:
                q = glm.angleAxis(math.pi/2, glm.vec3(0, 1, 0)) * q

            if part in ['lHip']:
                q = glm.angleAxis(math.pi/48, glm.vec3(1, 0, 0)) * q
            if part in ['rHip']:
                q = glm.angleAxis(-math.pi/48, glm.vec3(1, 0, 0)) * q

            v = basepose[i]
            motion[i] = q * v
        ret.append(motion)
    return ret



#生成したモーションのチェック
if __name__=='__main__':
    # 物理シミュレーションのGUIモードで接続
    physicsClient = p.connect(p.GUI)  
    p.setAdditionalSearchPath(pybullet_data.getDataPath())


    # 環境初期化
    p.setGravity(0, 0, 0)
    # p.setGravity(0, 0,0)
    # リアルタイムで動かすならこれを有効に
    p.setRealTimeSimulation(0)  # 0なら手動stepSimulation()

    timestep = 0.01
    p.setTimeStep(timestep)
    #モデル読み込み
    floor = p.loadURDF('plane.urdf')

    atlas_path = 'pybullet_robots/data/atlas_add_footsensor/atlas_v4_with_multisense.urdf'

    atlas1 = p.loadURDF(atlas_path, [0, 0, 1.2])

    # カメラの初期位置
    # Yaw:Z軸方向に右ねじの回転系
    # Pitch：90(真下)～-90(真上)
    p.resetDebugVisualizerCamera(cameraDistance=2,
                                 cameraYaw=45,
                                 cameraPitch=0,
                                 cameraTargetPosition=[0.5,0,0.8])
    
       


    init_pose_reset(atlas1, physicsClient)
    motions = ozigi()
    n_motions = len(motions)

    for i in range(n_motions):
        motion = motions[i]
        set_pose(atlas1, motion, physicsClient)
        mot = get_pose(atlas1, physicsClient)
        if i % 20 == 0:
            motion_viewer(mot, physicsClient, glm.vec3(0, 1, 1))
        p.stepSimulation()
        time.sleep(timestep)

    init_pose_reset(atlas1, physicsClient)
    for i in range(300):
        motion = achiles()
        set_pose(atlas1, motion, physicsClient)
        mot = get_pose(atlas1, physicsClient)
        if i % 20 == 0:
            motion_viewer(mot, physicsClient, glm.vec3(0, 1, 1))
        p.stepSimulation()
        time.sleep(timestep)

    init_pose_reset(atlas1, physicsClient)
    motions = wark()
    n_motions = len(motions)

    for i in range(n_motions):
        motion = motions[i]
        set_pose(atlas1, motion, physicsClient)
        mot = get_pose(atlas1, physicsClient)
        if i % 20 == 0:
            motion_viewer(mot, physicsClient, glm.vec3(0, 1, 1))
        p.stepSimulation()
        time.sleep(timestep)

    motions = standing()
    n_motions = len(motions)

    for i in range(n_motions):
        motion = motions[i]
        set_pose(atlas1, motion, physicsClient)
        mot = get_pose(atlas1, physicsClient)
        if i % 20 == 0:
            motion_viewer(mot, physicsClient, glm.vec3(0, 1, 1))
        p.stepSimulation()
        time.sleep(timestep)


    counter = 0
    while True:
        mot = get_pose(atlas1, physicsClient)
        if counter % 20 == 0:
            motion_viewer(mot, physicsClient, glm.vec3(0, 1, 1))

        if counter % 100 ==  0:
            print(counter)

        counter += 1
        time.sleep(timestep)
   
    p.disconnect()