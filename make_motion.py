#お辞儀やしゃがみなどの簡単なモーションを作成する
import math
import time

import glm
import numpy as np

import pybullet as p

from motion_controller import motion_viewer


vecName2idx = {'foot':0, 'sune':1, 'momo':2, 'body':3}


def ozigi():
    motions = []
    #1秒直立
    #3秒お辞儀
    #1秒待機
    #3秒で頭を上げる
    #2秒待機
    basepose = {0:glm.vec3(0, 0, 1), 1:glm.vec3(0, 0, 1), 2:glm.vec3(0, 0, 1), 3:glm.vec3(0, 0, 1)}

    #1秒直立
    for i in range(100):
        motions.append(basepose)

    #3秒でお辞儀
    for i in range(300):
        theta = (math.pi / 3) /300 * i

        motion = {0:basepose[0], 1:basepose[1], 2:basepose[2],
                  3:glm.vec3(math.sin(theta), 0, math.cos(theta))}
        motions.append(motion)
        
    #1秒キープ
    for i in range(100):
        theta = (math.pi / 3)
        motion = {0:basepose[0], 1:basepose[1], 2:basepose[2],
                  3:glm.vec3(math.sin(theta), 0, math.cos(theta))}
        motions.append(motion)

    #3秒で頭を上げる
    for i in range(300):
        theta = (math.pi / 3) - (math.pi / 3) /300 * i
        motion = {0:basepose[0], 1:basepose[1], 2:basepose[2],
                  3:glm.vec3(math.sin(theta), 0, math.cos(theta))}
        motions.append(motion)

    #1秒直立
    for i in range(100):
        motions.append(basepose)

    return motions



#生成したモーションのチェック
if __name__=='__main__':
    # 物理シミュレーションのGUIモードで接続
    physicsClient = p.connect(p.GUI)  # または p.DIRECT で非表示モード

    # 環境初期化
    p.setGravity(0, 0, -9.8)
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
    

    motions = ozigi()
    n_motions = len(motions)

    for i in range(n_motions):
        motion = motions[i]
        if i % 20 == 0:
            motion_viewer(motion, physicsClient, glm.vec3(0, 0.3, 0.3))
        p.stepSimulation()
        time.sleep(timestep)
    p.disconnect()