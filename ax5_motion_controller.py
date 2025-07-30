#####################################
#Atlasの制御を行うための実装
#Atlasの姿勢を初期化する
#モーションベクトルから各関節を動かす
#現在の各リンク座標から実機でもモーションベクトルを作成する
#####################################
import math

import glm
import numpy as np

import pybullet as p

# アトラスのジョイント、リンクにアクセスするためのインデックス一覧
jointName2idx ={
    #左腕
    'l_arm_shz':3, 'l_arm_shx':4, 'l_arm_ely':5, 'l_arm_elx':6, 
    'l_arm_wry':7, 'l_arm_wrx':8, 'l_arm_wry2':9, 

    #右腕
    'r_arm_shz':11, 'r_arm_shx':12, 'r_arm_ely':13, 'r_arm_elx':14, 
    'r_arm_wry':15, 'r_arm_wrx':16, 'r_arm_wry2':17, 
    #左足
    'l_leg_hpz':18, 'l_leg_hpx':19, 'l_leg_hpy':20, 'l_leg_kny':21, 
    'l_leg_aky':22, 'l_leg_akx':23, 
    #右足
    'r_leg_hpz':26, 'r_leg_hpx':27, 'r_leg_hpy':28, 'r_leg_kny':29, 
    'r_leg_aky':30, 'r_leg_akx':31}

joint_range_of_motion ={
'l_arm_shz' :( -1.5708 , 0.785398)   ,'l_arm_shx' :( -1.5708 , 1.5708)
,'l_arm_ely' :( 0 , 3.14159)         ,'l_arm_elx' :( 0 , 2.35619)
,'l_arm_wry' :( 0 , 3.14159)         ,'l_arm_wrx' :( -1.1781 , 1.1781)
,'l_arm_wry2':( -0.001 , 0.001)      

,'r_arm_shz' :( -0.785398 , 1.5708)  ,'r_arm_shx' :( -1.5708 , 1.5708)
,'r_arm_ely' :( 0 , 3.14159)         ,'r_arm_elx' :( -2.35619 , 0)
,'r_arm_wry' :( 0 , 3.14159)         ,'r_arm_wrx' :( -1.1781 , 1.1781)
,'r_arm_wry2':( -0.001 , 0.001)

,'l_leg_hpz' :( -0.174358 , 0.786794),'l_leg_hpx' :( -0.523599 , 0.523599)
,'l_leg_hpy' :( -1.61234 , 0.65764)  ,'l_leg_kny' :( 0 , 2.35637)
,'l_leg_aky' :( -1 , 0.7)            ,'l_leg_akx' :( -0.8 , 0.8)

,'r_leg_hpz' :( -0.786794 , 0.174358),'r_leg_hpx' :( -0.523599 , 0.523599)
,'r_leg_hpy' :( -1.61234 , 0.65764)  ,'r_leg_kny' :( 0 , 2.35637)
,'r_leg_aky' :( -1 , 0.7)            ,'r_leg_akx' :( -0.8 , 0.8)}

linkName2idx = {
    #体幹
    'ltorso': 0, 'mtorso': 1, 'utorso': 2, 'head': 10, 

    #左腕
    'l_clav': 3, 'l_scap': 4, 'l_uarm': 5, 
    'l_larm': 6, 'l_ufarm': 7, 'l_lfarm': 8, 'l_hand': 9, 
    
    #右腕
    'r_clav': 11, 'r_scap': 12, 'r_uarm': 13, 
    'r_larm': 14, 'r_ufarm': 15, 'r_lfarm': 16, 'r_hand': 17, 

    #左足
    'l_uglut': 18, 'l_lglut': 19, 'l_uleg': 20, 'l_lleg': 21, 'l_talus': 22, 
    'l_foot': 23, 'l_foot_toe_sensor': 24, 'l_foot_heel_sensor': 25, 
    
    #右足
    'r_uglut': 26, 'r_lglut': 27, 'r_uleg': 28, 'r_lleg': 29, 'r_talus': 30, 
    'r_foot': 31, 'r_foot_toe_sensor': 32, 'r_foot_heel_sensor': 33, 
}

vecName2idx = {'body':0, 
               'Lmomo':1, 'Lsune':2, 'Lfoot':3,
               'Rmomo':4, 'Rsune':5, 'Rfoot':6}


def init_pose(atlas_id, physicsClient):
    #atlasの姿勢の初期化　肩を閉じるなど
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_shx'],
                            controlMode=p.POSITION_CONTROL, targetPosition= -math.pi/2
                            , physicsClientId=physicsClient)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_arm_shx'],
                            controlMode=p.POSITION_CONTROL, targetPosition=  math.pi/2
                            , physicsClientId=physicsClient)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_elx'],
                            controlMode=p.POSITION_CONTROL, targetPosition= 0
                            , physicsClientId=physicsClient)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_arm_elx'],
                            controlMode=p.POSITION_CONTROL, targetPosition= 0
                            , physicsClientId=physicsClient)

def init_pose_reset(atlas_id, physicsClient):
    #atlasの姿勢の初期化　肩を閉じるなど
    #Resetなので反動はないはず
    p.resetJointState(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_shx'],
                            targetValue= -math.pi/2, physicsClientId=physicsClient)
    p.resetJointState(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_arm_shx'],
                            targetValue=  math.pi/2, physicsClientId=physicsClient)
    p.resetJointState(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_elx'],
                            targetValue= 0, physicsClientId=physicsClient)
    p.resetJointState(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_arm_elx'],
                             targetValue= 0, physicsClientId=physicsClient)


def set_pose(atlas_id, motion, physicsClient):
    def angle_between_2vec_onXZ_YZ(A:glm.vec3, B:glm.vec3): #xz,yx平面においてA→Bの角度差を算出(x軸基準)
        A_x = A[0]
        A_y = A[1]
        A_z = A[2]
        B_x = B[0]
        B_y = B[1]
        B_z = B[2]
        #XZ平面
        thetaA_xz = math.atan2(A_z, A_x)
        thetaB_xz = math.atan2(B_z, B_x)
        delta_xz = thetaB_xz - thetaA_xz
        delta_xz = ((delta_xz + math.pi) % (2 * math.pi)) - math.pi

        #YZ平面
        thetaA_yz = math.atan2(A_z, A_y)
        thetaB_yz = math.atan2(B_z, B_y)
        delta_yz = thetaB_yz - thetaA_yz
        delta_yz = ((delta_yz + math.pi) % (2 * math.pi)) - math.pi

        return delta_xz, delta_yz
    

    #各体勢ベクトル
    vec_body = motion[vecName2idx['body']]
    vec_L_momo = motion[vecName2idx['Lmomo']]
    vec_L_sune = motion[vecName2idx['Lsune']]
    vec_L_foot = motion[vecName2idx['Lfoot']]
    vec_R_momo = motion[vecName2idx['Rmomo']]
    vec_R_sune = motion[vecName2idx['Rsune']]
    vec_R_foot = motion[vecName2idx['Rfoot']]
    
    
    #各関節についてベクトルどうしのなす角を計算
    #XZ平面のでの回転の向きとY軸の右ねじの関係上、関節角をセットするときには正負逆転が必要
    #YZ平面においては回転の向きとX軸の右ねじが一致するので回転は不要

    #左
    theta_L_hip_ardY, theta_L_hip_ardX = angle_between_2vec_onXZ_YZ(-vec_body, vec_L_momo)
    theta_L_hip_ardY = -theta_L_hip_ardY

    theta_L_knee_ardY, _               = angle_between_2vec_onXZ_YZ(vec_L_momo, vec_L_sune)
    theta_L_knee_ardY = -theta_L_knee_ardY

    theta_L_ankl_ardY, theta_L_ankl_ardX = angle_between_2vec_onXZ_YZ(vec_L_sune, vec_L_foot)
    theta_L_ankl_ardY = -theta_L_ankl_ardY

    #右
    theta_R_hip_ardY, theta_R_hip_ardX = angle_between_2vec_onXZ_YZ(-vec_body, vec_R_momo)
    theta_R_hip_ardY = -theta_R_hip_ardY

    theta_R_knee_ardY, _               = angle_between_2vec_onXZ_YZ(vec_R_momo, vec_R_sune)
    theta_R_knee_ardY = -theta_R_knee_ardY

    theta_R_ankr_ardY, theta_R_ankr_ardX = angle_between_2vec_onXZ_YZ(vec_R_sune, vec_R_foot)
    theta_R_ankr_ardY = -theta_R_ankr_ardY

    # print(f'angles... ancle:{theta_ancle:.3f} knee:{theta_knee:.3f} hip:{theta_hip:.3f}')

    #関節角度のセット

    #hip
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_leg_hpy'],
                            controlMode=p.POSITION_CONTROL, targetPosition= theta_L_hip_ardY
                            , physicsClientId=physicsClient)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_leg_hpy'],
                            controlMode=p.POSITION_CONTROL, targetPosition= theta_R_hip_ardY
                            , physicsClientId=physicsClient)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_leg_hpx'],
                            controlMode=p.POSITION_CONTROL, targetPosition= theta_L_hip_ardX
                            , physicsClientId=physicsClient)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_leg_hpx'],
                            controlMode=p.POSITION_CONTROL, targetPosition= theta_R_hip_ardX
                            , physicsClientId=physicsClient)
    
    #knee
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_leg_kny'],
                            controlMode=p.POSITION_CONTROL, targetPosition= theta_L_knee_ardY
                            , physicsClientId=physicsClient)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_leg_kny'],
                            controlMode=p.POSITION_CONTROL, targetPosition= theta_R_knee_ardY
                            , physicsClientId=physicsClient)
    
    #ankle
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_leg_aky'],
                            controlMode=p.POSITION_CONTROL, targetPosition= theta_L_ankl_ardY
                            , physicsClientId=physicsClient)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_leg_aky'],
                            controlMode=p.POSITION_CONTROL, targetPosition= theta_R_ankr_ardY
                            , physicsClientId=physicsClient)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_leg_akx'],
                            controlMode=p.POSITION_CONTROL, targetPosition= theta_L_ankl_ardX
                            , physicsClientId=physicsClient)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_leg_akx'],
                            controlMode=p.POSITION_CONTROL, targetPosition= theta_R_ankr_ardX
                            , physicsClientId=physicsClient)
    
def get_pose(atlas_id, physicsClient):
    '''
    現在のAtlasのリンク座標から体の向きベクトルを取得する
    '''
    ret = {}
    #各リンク座標の取得
    num_joints = p.getNumJoints(atlas_id, physicsClientId=physicsClient)

    positions = {}
    for link_index in range(num_joints):
        info = p.getJointInfo(atlas_id, link_index, physicsClientId=physicsClient)
        link_state = p.getLinkState(atlas_id, link_index, computeForwardKinematics=True
                                    , physicsClientId=physicsClient)
        pos = link_state[0]
#       orn = link_state[1]
        link_name = info[12].decode("utf-8")  # 名前（bytes → str）
#       joint_name = info[1].decode("utf-8")
        position = glm.vec3(pos[0], pos[1], pos[2])
        positions[linkName2idx[link_name]] = position


    #体幹
    v_body = positions[linkName2idx['utorso']] - positions[linkName2idx['ltorso']]

    #股関節から膝
    v_r_hip2knee = positions[linkName2idx['r_uleg']] - positions[linkName2idx['r_lglut']]
    v_l_hip2knee = positions[linkName2idx['l_uleg']] - positions[linkName2idx['l_lglut']]
    
    #膝からかかと
    v_r_knee2heel = positions[linkName2idx['r_talus']] - positions[linkName2idx['r_lleg']]
    v_l_knee2heel = positions[linkName2idx['l_talus']] - positions[linkName2idx['l_lleg']]

    #両足
     #かかとからつま先
    v_r_heel2toe = positions[linkName2idx['r_foot_toe_sensor']] - \
                    positions[linkName2idx['r_foot_heel_sensor']]
    v_l_heel2toe = positions[linkName2idx['l_foot_toe_sensor']] - \
                    positions[linkName2idx['l_foot_heel_sensor']]
     #Y軸にそって90度回転：法線方向
     #※ロボットがZ軸に沿って回転する場合はここがエラーの原因になりうる
    q = glm.angleAxis(math.pi/2, glm.vec3(0, 1, 0))
     #足の裏の法線ベクトル
    v_r_foot = q * v_r_heel2toe
    v_l_foot = q * v_l_heel2toe

    #正規化
    ret[vecName2idx['body']] =  glm.normalize(v_body)
    ret[vecName2idx['Lmomo']] =  glm.normalize(v_l_hip2knee)
    ret[vecName2idx['Lsune']] =  glm.normalize(v_l_knee2heel)
    ret[vecName2idx['Lfoot']] =  glm.normalize(v_l_foot)
    ret[vecName2idx['Rmomo']] =  glm.normalize(v_r_hip2knee)
    ret[vecName2idx['Rsune']] =  glm.normalize(v_r_knee2heel)
    ret[vecName2idx['Rfoot']] =  glm.normalize(v_r_foot)
    return ret

def motion_viewer(motions:dict, physicsClient, body_occurrence_point:glm.vec3=glm.vec3(0,0,0)):
    '''
    モーションデータの可視化
    '''
    p_base = body_occurrence_point

    #足の向き
    q = glm.angleAxis(-math.pi/2, glm.vec3(0, 1, 0))


    v_body = motions[vecName2idx['body']]
    v_Lmomo = motions[vecName2idx['Lmomo']]
    v_Lsune = motions[vecName2idx['Lsune']]
    v_Lfoot = q * motions[vecName2idx['Lfoot']]
    v_Rmomo = motions[vecName2idx['Rmomo']]
    v_Rsune = motions[vecName2idx['Rsune']]
    v_Rfoot = q * motions[vecName2idx['Rfoot']]

    p_neck = p_base + v_body * 1.
    p_l_hip  = p_base + glm.vec3(0,  0.2, 0)
    p_r_hip  = p_base + glm.vec3(0, -0.2, 0)
    p_l_knee = p_l_hip + v_Lmomo * 0.5
    p_r_knee = p_r_hip + v_Rmomo * 0.5
    p_l_heel = p_l_knee + v_Lsune * 0.5
    p_r_heel = p_r_knee + v_Rsune * 0.5
    p_l_tue = p_l_heel + v_Lfoot * 0.2
    p_r_tue = p_r_heel + v_Rfoot * 0.2
    
    #直線の描写
    def gV2ls(vec:glm.vec3):
        return [vec.x, vec.y, vec.z]
    
    p.addUserDebugLine(gV2ls(p_base), gV2ls(p_neck), lineColorRGB=[1, 0, 0], lifeTime=0.99
                       , physicsClientId=physicsClient)

    p.addUserDebugLine(gV2ls(p_l_hip), gV2ls(p_l_knee), lineColorRGB=[0, 1, 0], lifeTime=0.99
                       , physicsClientId=physicsClient)
    p.addUserDebugLine(gV2ls(p_l_knee), gV2ls(p_l_heel), lineColorRGB=[0, 1, 0], lifeTime=0.99
                       , physicsClientId=physicsClient)
    p.addUserDebugLine(gV2ls(p_l_heel), gV2ls(p_l_tue), lineColorRGB=[0, 1, 0], lifeTime=0.99
                       , physicsClientId=physicsClient)

    p.addUserDebugLine(gV2ls(p_r_hip), gV2ls(p_r_knee), lineColorRGB=[1, 0, 0], lifeTime=0.99
                       , physicsClientId=physicsClient)
    p.addUserDebugLine(gV2ls(p_r_knee), gV2ls(p_r_heel), lineColorRGB=[1, 0, 0], lifeTime=0.99
                       , physicsClientId=physicsClient)
    p.addUserDebugLine(gV2ls(p_r_heel), gV2ls(p_r_tue), lineColorRGB=[1, 0, 0], lifeTime=0.99
                       , physicsClientId=physicsClient)