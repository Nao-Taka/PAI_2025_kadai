#モーションデータをアトラスのモーションに
#アトラスのモーションをモーションデータに
#相互に変換する

import time

import numpy as np
import pybullet as p
import pybullet_data

import glm

ROTATE = 2* np.pi

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

#今のところ各クオータニオンへのアクセスは辞書に対する部位名
#quatName2idx = {'Body':0, 'rShldr':1, 'rElbow':2, 'lShldr':3, 'lElbow':4, 
#               'rHip':5, 'rKnee':6, 'rAnkle':7, 'lHip':8, 'lKnee':9, 'lAnkle':10}


def set_motion(atlas_id, motions:dict):
    '''
    あるフレームのモーションをアトラスに設定する
    250725時点ではモーションキャプチャの座標から、物理エンジンへの座標変換の吸収はこちらで行っている
    本来はモーションキャプチャのほうで吸収するのが望ましい、、、
    '''

    #左肩
    mt = motions['lShldr']
    euler_rad = glm.eulerAngles(mt)
    x_rad = euler_rad[0]
    y_rad = euler_rad[1]
    z_rad = euler_rad[2]
    print(f'{x_rad} {y_rad} {z_rad}')
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_shz'],
                            controlMode=p.POSITION_CONTROL, targetPosition=y_rad)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_shx'],
                            controlMode=p.POSITION_CONTROL, targetPosition=z_rad)
    
    #肘の計算
    #肩ベクトルの延長と肘ベクトルのなす角をl_arm_elxに設定
    #その後、プランの肩ベクトルｰ肘ベクトル、実際の肩ベクトル-肘ベクトルによってなす面の
    #角度だけl_arm_elyを補正
    #1つ目のなす角が一定値より小さい場合はl_arm_elyの変更は行わない
    mt:glm.quat = motions['rElbow']
    w = mt.w
    theta = 2 * np.arccos(np.clip(w, -1., 1.))
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_elx'],
                            controlMode=p.POSITION_CONTROL, targetPosition=theta)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_ely'],
                            controlMode=p.POSITION_CONTROL, targetPosition=ROTATE/4)
    
    print(theta)







def quaternion_viewer(motions:dict):
    '''
    クオータニオンの可視化をする
    主にデバッグ用
    '''

    v_body = glm.vec3(0, 0, 1)
    v_b_neck = glm.vec3(0, 0, 0.5)
    v_n_lshd = glm.vec3(0, 0.2, 0)
    v_n_rshd = glm.vec3(0, -0.2, 0)

    #肩から肘
    q_lshd = motions['lShldr']
    q_rshd = motions['rShldr']
    v_lshd_lelb = q_lshd * (2.5 * v_n_lshd)
    v_rshd_relb = q_rshd * (2.5 * v_n_rshd)

    #肘から手
    q_lelb = motions['lElbow']
    q_relb = motions['rElbow']
    v_lelb_lhnd = q_lelb * v_lshd_lelb
    v_relb_rhnd = q_relb * v_rshd_relb

    #骨盤ベースから股関節
    v_b_lhip = glm.vec3(0, 0.2, 0)
    v_b_rhip = glm.vec3(0, -0.2, 0)

    #股関節から膝
    v_lhip_lkne = glm.vec3(0, 0, -0.5)
    v_rhip_rkne = glm.vec3(0, 0, -0.5)
    q_lhip = motions['lHip']
    q_rhip = motions['rHip']
    v_lhip_lkne = q_lhip * v_lhip_lkne
    v_rhip_rkne = q_rhip * v_rhip_rkne

    #膝から足首
    q_lKnee = motions['lKnee']
    q_rKnee = motions['rKnee']
    v_lkne_lank = q_lKnee * v_lhip_lkne
    v_rkne_rank = q_rKnee * v_rhip_rkne

    #足首からつま先
    q_lank = motions['lAnkle']
    q_rank = motions['rAnkle']
    v_lank_ltue = q_lank * (v_lkne_lank / 3)
    v_rank_rtue = q_rank * (v_rkne_rank / 3)

    #各座標の取得
    p_body = v_body
    p_neck = p_body + v_b_neck
     #腕
    p_lshd = p_neck + v_n_lshd
    p_lelb = p_lshd + v_lshd_lelb
    p_lhnd = p_lelb + v_lelb_lhnd
    p_rshd = p_neck + v_n_rshd
    p_relb = p_rshd + v_rshd_relb
    p_rhnd = p_relb + v_relb_rhnd
     #足
    p_lhip = p_body + v_b_lhip
    p_lkne = p_lhip + v_lhip_lkne
    p_lank = p_lkne + v_lkne_lank
    p_ltue = p_lank + v_lank_ltue
    p_rhip = p_body + v_b_rhip
    p_rkne = p_rhip + v_rhip_rkne
    p_rank = p_rkne + v_rkne_rank
    p_rtue = p_rank + v_rank_rtue
    def gV2ls(vec:glm.vec3):
        return [vec.x, vec.y, vec.z]

    #直線の描写
    p.addUserDebugLine(gV2ls(p_body), gV2ls(p_neck), lineColorRGB=[1, 0, 0], lifeTime=0.99)
     #左半身
    p.addUserDebugLine(gV2ls(p_neck), gV2ls(p_lshd), lineColorRGB=[1, 0, 0], lifeTime=0.99)
    p.addUserDebugLine(gV2ls(p_lshd), gV2ls(p_lelb), lineColorRGB=[1, 0, 0], lifeTime=0.99)
    p.addUserDebugLine(gV2ls(p_lelb), gV2ls(p_lhnd), lineColorRGB=[1, 0, 0], lifeTime=0.99)
    p.addUserDebugLine(gV2ls(p_body), gV2ls(p_lhip), lineColorRGB=[1, 0, 0], lifeTime=0.99)
    p.addUserDebugLine(gV2ls(p_lhip), gV2ls(p_lkne), lineColorRGB=[1, 0, 0], lifeTime=0.99)
    p.addUserDebugLine(gV2ls(p_lkne), gV2ls(p_lank), lineColorRGB=[1, 0, 0], lifeTime=0.99)
    p.addUserDebugLine(gV2ls(p_lank), gV2ls(p_ltue), lineColorRGB=[1, 0, 0], lifeTime=0.99)
     #右半身
    p.addUserDebugLine(gV2ls(p_neck), gV2ls(p_rshd), lineColorRGB=[0, 1, 0], lifeTime=0.99)
    p.addUserDebugLine(gV2ls(p_rshd), gV2ls(p_relb), lineColorRGB=[0, 1, 0], lifeTime=0.99)
    p.addUserDebugLine(gV2ls(p_relb), gV2ls(p_rhnd), lineColorRGB=[0, 1, 0], lifeTime=0.99)
    p.addUserDebugLine(gV2ls(p_body), gV2ls(p_rhip), lineColorRGB=[0, 1, 0], lifeTime=0.99)
    p.addUserDebugLine(gV2ls(p_rhip), gV2ls(p_rkne), lineColorRGB=[0, 1, 0], lifeTime=0.99)
    p.addUserDebugLine(gV2ls(p_rkne), gV2ls(p_rank), lineColorRGB=[0, 1, 0], lifeTime=0.99)
    p.addUserDebugLine(gV2ls(p_rank), gV2ls(p_rtue), lineColorRGB=[0, 1, 0], lifeTime=0.99)




    


