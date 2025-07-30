#モーションデータをアトラスのモーションに
#アトラスのモーションをモーションデータに
#相互に変換する
import math
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar,minimize

import glm
import pybullet as p
import pybullet_data

ROTATE = 2* np.pi

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

#各クオータニオンへのアクセスに使うインデックスを指定する辞書
#bvh2absmotと一致する必要あり
quatName2idx = {'body':0, 'rShldr':1, 'rElbow':2, 'lShldr':3, 'lElbow':4, 
               'rHip':5, 'rKnee':6, 'rAnkle':7, 'lHip':8, 'lKnee':9, 'lAnkle':10}

def set_motion(atlas_id, motions:dict):
    '''
    あるフレームのモーションをアトラスに設定する
    250725時点ではモーションキャプチャの座標から、物理エンジンへの座標変換の吸収はこちらで行っている
    本来はモーションキャプチャのほうで吸収するのが望ましい、、、
    →吸収しました

    肩関節：仰角と向きをクオータニオンから計算
    その後肘関節の軸方向からの傾きをクオーテーションから取得
    最後に肘からの向きベクトルからのずれを計算して補正

    股関節：
    太ももを軸とした回転補正は解析的に求まるはずだが、わからないので、
    z軸を0-90で回転させたときに膝座標が合うようにx、yを回転、その中で最も誤差が小さいものを正解とする方針に
    '''
    base_px = glm.vec3( 1,  0,  0)
    base_py = glm.vec3( 0,  1,  0)
    base_pz = glm.vec3( 0,  0,  1)
    base_nx = glm.vec3(-1,  0,  0)
    base_ny = glm.vec3( 0, -1,  0)
    base_nz = glm.vec3( 0,  0, -1)

    #各向きのベクトルからクオータニオンを計算する
    def quat_from_two_vectors(a: glm.vec3, b: glm.vec3) -> glm.quat: #A to B(x,y,z,w)
        a = glm.normalize(a)
        b = glm.normalize(b)
        dot = glm.dot(a, b)
        if dot >= 1.0:
            return glm.quat(1, 0, 0, 0)  # 単位クオータニオン（回転なし）
        elif dot <= -1.0:
            # 真逆方向 → 任意の直交軸で180度回転
            orthogonal = glm.vec3(1, 0, 0) if abs(a.x) < 0.9 else glm.vec3(0, 1, 0)
            axis = glm.normalize(glm.cross(a, orthogonal))
            return glm.angleAxis(math.pi, axis)
        else:
            axis = glm.normalize(glm.cross(a, b))
            angle = math.acos(dot)
            return glm.angleAxis(angle, axis)
        
    #クオータニオンによってベクトルがAからA’に変形されるとき、AとA'がなす角を求める
    def quat_to_ax_theta(quaternion:glm.quat):
        w = quaternion.w
        theta = 2 * np.arccos(np.clip(w, -1., 1.))
        return theta

    #ベクトルAとBがなす角
    def theta_a_b(a:glm.vec3, b:glm.vec3):
        # 内積と長さ
        dot_product = glm.dot(a, b)
        lengths_product = glm.length(a) * glm.length(b)
        # 安全に acos に入れる（丸め誤差で ±1 を超えるとエラーになる）
        cos_theta = glm.clamp(dot_product / lengths_product, -1.0, 1.0)
        # ラジアンで角度を求める
        theta_rad = math.acos(cos_theta)
        return theta_rad


    #ベクトルAと任意の平面がなす角(平面は法線ベクトルで与える)
    def theta_a_Plane(a:glm.vec3, ax:glm.vec3):
        A_norm = glm.normalize(a)
        #xy平面の場合
        N_norm = glm.normalize(ax)
        # ベクトルAと法線Nのなす角 φ
        dot = glm.dot(A_norm, N_norm)
        dot = glm.clamp(dot, -1.0, 1.0)  # 安全のため
        phi = math.acos(dot)  # in radians
        # 平面とベクトルAのなす角 θ
        theta = math.pi / 2 - phi
        return theta
    
    #平面αと平面βがなす角
    #ベクトルa, b　ベクトルc, dが形成する平面が形成する角
    def theta_plateA_plateB(a:glm.vec3, b:glm.vec3, c:glm.vec3, d:glm.vec3):
        #平面が形成されない場合None
        t_a_b = theta_a_b(a, b)
        t_c_d = theta_a_b(c, d)
        minimam_theta = 0.015
        maximam_theta = 3.13
        if t_a_b < minimam_theta or t_c_d < minimam_theta \
           or t_a_b > maximam_theta or t_c_d > maximam_theta:
            return None
        alpha_norm = glm.cross(a, b) 
        beta_norm = glm.cross(c, d)
        #2つの平面の法線ベクトルがなす角
        theta = theta_a_b(alpha_norm, beta_norm)
        return glm.normalize(theta)

    #ベクトルA,Bが与えられたとき、Aを(x, y, z)任意の軸axで回転させて最もBに近づくthetaとそのA’を求める
    # theta, A'
    def find_best_rotation_angle(A: glm.vec3, B: glm.vec3, axis: glm.vec3):
        # 正規化
        A = glm.normalize(A)
        B = glm.normalize(B)
        axis = glm.normalize(axis)

        # 評価関数：Aをθ回転させたときのBとのなす角（cosθのマイナス）
        def objective(theta_rad):
            q = glm.angleAxis(theta_rad, axis)
            A_rot = q * A
            cos_angle = glm.dot(glm.normalize(A_rot), B)
            return -cos_angle  # 最大化したいので符号反転

        # 最適化：theta ∈ [-π, π]
        result = minimize_scalar(objective, bounds=(-math.pi, math.pi), method='bounded')

        best_theta = result.x
        best_q = glm.angleAxis(best_theta, axis)
        best_A_rot = best_q * A

        return best_theta, best_A_rot

    #ベクトルの外積を求める
    def cross_norm(a:glm.vec3, b:glm.vec3):
        return glm.normalize(glm.cross(a, b))
    
    #ベクトルAを任意のベクトルNを軸としてThetaだけ回転させる
    def rotate_vector_around_axis(vec: glm.vec3, axis: glm.vec3, theta_rad: float) -> glm.vec3:
        """
        任意の軸 `axis` を中心に、ベクトル `vec` を `theta_rad` 回転させた結果を返す。
        Parameters:
            vec (glm.vec3): 回転させたい元のベクトル
            axis (glm.vec3): 回転軸（任意方向、正規化は自動で行われる）
            theta_rad (float): 回転角度（ラジアン）

        Returns:
            glm.vec3: 回転後のベクトル
        """
        axis = glm.normalize(axis)
        q = glm.angleAxis(theta_rad, axis)
        return q * vec
    
    
    # #上半身
    # '''
    # クオータニオンから理想の方向ベクトルを計算
    # 肩からの向きが理想と近しくなるようにx軸、z軸を最適化
    # 肘のまげの角度はクオータニオンから計算
    # 肘の回転を理想と近しくなるように最適化
    # '''
    # #各クオータニオンの取得
    # quat_l_shld = motions[quatName2idx['lShldr']]
    # quat_r_shld = motions[quatName2idx['rShldr']]
    # quat_l_elbw = motions[quatName2idx['lElbow']]
    # quat_r_elbw = motions[quatName2idx['rElbow']]

    # ideal_vec_l_clavicle = base_py
    # ideal_vec_r_clavicle = base_ny
    # ideal_vec_l_humerus = quat_l_shld *ideal_vec_l_clavicle
    # ideal_vec_r_humerus = quat_r_shld *ideal_vec_r_clavicle
    # ideal_vec_l_radius = quat_l_elbw * ideal_vec_l_humerus
    # ideal_vec_r_radius = quat_r_elbw * ideal_vec_r_humerus

    # #肘のなす角
    # dig_l_elbow =  quat_to_ax_theta(quat_l_elbw)
    # dig_r_elbow = -quat_to_ax_theta(quat_r_elbw)

    # #ロボットの上腕の向きの初期値
    # real_vec_l_humerus = base_py
    # real_vec_r_humerus = base_ny

    # #ベースの軸を規定するためのx軸方向のベクトル
    # real_vec_l_humerus_sup = base_px
    # real_vec_r_humerus_sup = base_nx

    # #l/r_arm_shx'　肩関節x軸方向の最適化
    # #real_vec_l/r_humerusをx軸方向に回転させてideal___との距離を最小にする

    # def objective(theta_rads, real_vec_humerus, ideal_vec_humerus):
    #     rad1, rad2 = theta_rads
    #     rotated_vec = rotate_vector_around_axis(real_vec_humerus, base_px, rad1)
    #     rotated_vec = rotate_vector_around_axis(rotated_vec, base_pz, rad2)
    #     return -glm.dot(glm.normalize(rotated_vec), glm.normalize(ideal_vec_humerus))
    # x0 = [0.,0.]
    # bounds = [joint_range_of_motion['l_arm_shx'], joint_range_of_motion['l_arm_shz']]
    # args = (real_vec_l_humerus, ideal_vec_l_humerus)

    # results = minimize(fun=objective, x0=x0, method='L-BFGS-B', bounds=bounds, args=args)
    # l_arm_shx, l_arm_shz = results.x

    # x0 = [0.,0.]
    # bounds = [joint_range_of_motion['r_arm_shx'], joint_range_of_motion['r_arm_shz']]
    # args = (real_vec_r_humerus, ideal_vec_r_humerus)

    # results = minimize(fun=objective, x0=x0, method='L-BFGS-B', bounds=bounds, args=args)
    # r_arm_shx, r_arm_shz = results.x

    # p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_shz'],
    #                          controlMode=p.POSITION_CONTROL, targetPosition=l_arm_shz)
    # p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_shx'],
    #                          controlMode=p.POSITION_CONTROL, targetPosition=l_arm_shx)
    # p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_elx'],
    #                          controlMode=p.POSITION_CONTROL, targetPosition=dig_l_elbow)
    

    # p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_arm_shz'],
    #                          controlMode=p.POSITION_CONTROL, targetPosition=r_arm_shz)
    # p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_arm_shx'],
    #                          controlMode=p.POSITION_CONTROL, targetPosition=r_arm_shx)
    # p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_arm_elx'],
    #                          controlMode=p.POSITION_CONTROL, targetPosition=dig_r_elbow)



############################################################################
#上：最適化による導出　荒ぶる
#下；計算による導出　なんとなく微妙
############################################################################

    

    #左肩
    quat = motions[quatName2idx['lShldr']]
    quated_vec_Uarm = quat * base_py
    quated_vec_Uarm_supX = quat * base_px
    #XY平面となす角
    theta_XY = theta_a_Plane(quated_vec_Uarm, base_pz)
    #z軸方向の回転
    theta_Zax, _ = find_best_rotation_angle(base_py, quated_vec_Uarm, base_pz)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_shz'],
                            controlMode=p.POSITION_CONTROL, targetPosition=theta_Zax)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_shx'],
                            controlMode=p.POSITION_CONTROL, targetPosition=theta_XY)

    #左肘
    '''
    quated_vec_Uarm_supXはAtlasから見た相対的な座標において、常に地面と平行かつ上腕のZ軸回転
    に連動して動く。
    そのため前腕はquated_vec_Uarm_supXを軸としてquated_vec_Uarmからtheta_hiziだけ回転する
    その際、理想のベクトルと基準となる回転ベクトルはquated_vec_Uarmを軸としてthetaだけずれている
    ので算出する
    '''
    quat = motions[quatName2idx['lElbow']]
    quated_vec_Farm = quat * quated_vec_Uarm
    theta_hizi = quat_to_ax_theta(quat)

    #quated_vec_Uarm_supXを軸としてquated_vec_Uarmからtheta_hiziだけ回転する
    baseVec = rotate_vector_around_axis(quated_vec_Uarm, quated_vec_Uarm_supX, theta_hizi)
    #理想とのずれを計算する
    theta, _ = find_best_rotation_angle(baseVec, quated_vec_Farm, quated_vec_Uarm)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_elx'],
                            controlMode=p.POSITION_CONTROL, targetPosition=theta_hizi)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_ely'],
                            controlMode=p.POSITION_CONTROL, targetPosition=theta)

    #右肩
    quat = motions[quatName2idx['rShldr']]
    quated_vec_Uarm = quat * base_ny
    quated_vec_Uarm_supX = quat * base_px
    #XY平面となす角
    theta_XY = theta_a_Plane(quated_vec_Uarm, base_nz)
    #z軸方向の回転
    theta_Zax, _ = find_best_rotation_angle(base_ny, quated_vec_Uarm, base_pz)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_arm_shz'],
                            controlMode=p.POSITION_CONTROL, targetPosition=theta_Zax)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_arm_shx'],
                            controlMode=p.POSITION_CONTROL, targetPosition=theta_XY)

    #右肘
    '''
    quated_vec_Uarm_supXはAtlasから見た相対的な座標において、常に地面と平行かつ上腕のZ軸回転
    に連動して動く。
    そのため前腕はquated_vec_Uarm_supXを軸としてquated_vec_Uarmからtheta_hiziだけ回転する
    その際、理想のベクトルと基準となる回転ベクトルはquated_vec_Uarmを軸としてthetaだけずれている
    ので算出する
    '''
    quat = motions[quatName2idx['rElbow']]
    quated_vec_Farm = quat * quated_vec_Uarm
    theta_hizi = quat_to_ax_theta(quat)

    #quated_vec_Uarm_supXを軸としてquated_vec_Uarmからtheta_hiziだけ回転する
    baseVec = rotate_vector_around_axis(quated_vec_Uarm, quated_vec_Uarm_supX, -theta_hizi)
    #理想とのずれを計算する
    theta, _ = find_best_rotation_angle(baseVec, quated_vec_Farm, quated_vec_Uarm)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_arm_elx'],
                            controlMode=p.POSITION_CONTROL, targetPosition= -theta_hizi)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_arm_ely'],
                            controlMode=p.POSITION_CONTROL, targetPosition=theta)
    

    #股関節
    quat_l_hip = motions[quatName2idx['lHip']]
    quat_r_hip = motions[quatName2idx['rHip']]
    quat_l_kne = motions[quatName2idx['lKnee']]
    quat_r_kne = motions[quatName2idx['rKnee']]

    #理想の相対座標
    ideal_vec_l_thigh = quat_l_hip * base_nz
    ideal_vec_r_thigh = quat_r_hip * base_nz

    ideal_vec_l_lowlg = quat_l_kne * ideal_vec_l_thigh
    ideal_vec_r_lowlg = quat_r_kne * ideal_vec_r_thigh

    #膝関節のなす角
    dig_l_knee = theta_a_b(ideal_vec_l_thigh, ideal_vec_l_lowlg)
    dig_r_knee = theta_a_b(ideal_vec_r_thigh, ideal_vec_r_lowlg)

    '''
    股関節のなす角を計算する
    股関節角度計算の問題点は膝の目標座標に対してｘ、ｙ、ｚが一意に定まらないことである
    ここでZを任意の値としたときに目標の座標に向かうために必要なｘ、ｙが一意に定まる
    ことを利用する

    目標の座標を
    。。。
    まずは簡単のためにｙ軸、ｚ軸にそった回転をかんがえることとする。。。


    '''

    #xz平面の写像からatanで計算
    l_leg_hpy = -math.atan2(ideal_vec_l_thigh.x, -ideal_vec_l_thigh.z)
    r_leg_hpy = -math.atan2(ideal_vec_r_thigh.x, -ideal_vec_r_thigh.z)

    # print(f'{r_leg_hpy} {ideal_vec_l_thigh.z}, {ideal_vec_l_thigh.x}')
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_leg_hpy'],
                            controlMode=p.POSITION_CONTROL, targetPosition= l_leg_hpy)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_leg_hpy'],
                            controlMode=p.POSITION_CONTROL, targetPosition= r_leg_hpy)
    

    #膝
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_leg_kny'],
                            controlMode=p.POSITION_CONTROL, targetPosition= dig_l_knee)
    p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['r_leg_kny'],
                            controlMode=p.POSITION_CONTROL, targetPosition= dig_r_knee)
    

    










    





    






    # #肩
    #  #左
    # mt = motions[quatName2idx['lShldr']]
    # euler_rad = glm.eulerAngles(mt)
    # x_rad = euler_rad[0]
    # y_rad = euler_rad[1]
    # z_rad = euler_rad[2]
    # print(f'{x_rad} {y_rad} {z_rad}')
    
    # #肘の計算
    # #肩ベクトルの延長と肘ベクトルのなす角をl_arm_elxに設定
    # #その後、プランの肩ベクトルｰ肘ベクトル、実際の肩ベクトル-肘ベクトルによってなす面の
    # #角度だけl_arm_elyを補正
    # #1つ目のなす角が一定値より小さい場合はl_arm_elyの変更は行わない
    # mt:glm.quat = motions[quatName2idx['rElbow']]
    # w = mt.w
    # theta = 2 * np.arccos(np.clip(w, -1., 1.))
    # p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_elx'],
    #                         controlMode=p.POSITION_CONTROL, targetPosition=theta)
    # p.setJointMotorControl2(bodyUniqueId=atlas_id, jointIndex=jointName2idx['l_arm_ely'],
    #                         controlMode=p.POSITION_CONTROL, targetPosition=ROTATE/4)
    


def get_motion(atlas_id):
    '''
    現在のアトラスの姿勢状態から
    中位層とやり取りをするために必要なクオータニオンを生成
    '''
    
    #各リンク座標の取得
    num_joints = p.getNumJoints(atlas_id)

    positions = {}
    for link_index in range(num_joints):
        info = p.getJointInfo(atlas_id, link_index)
        link_state = p.getLinkState(atlas_id, link_index, computeForwardKinematics=True)
        pos = link_state[0]
#       orn = link_state[1]
        link_name = info[12].decode("utf-8")  # 名前（bytes → str）
#       joint_name = info[1].decode("utf-8")
        position = glm.vec3(pos[0], pos[1], pos[2])
        positions[linkName2idx[link_name]] = position
    
    #各リンク座標から体の各部位の向きベクトルを計算
    #両腕
    v_rCol_rShl = positions[linkName2idx['r_clav']] - positions[linkName2idx['l_clav']]
    v_rShl_rArm = positions[linkName2idx['r_uarm']] - positions[linkName2idx['r_scap']]
    v_rArm_rHnd = positions[linkName2idx['r_ufarm']] - positions[linkName2idx['r_larm']]

    v_lCol_lShl = positions[linkName2idx['l_clav']] - positions[linkName2idx['r_clav']]
    v_lShl_lArm = positions[linkName2idx['l_uarm']] - positions[linkName2idx['l_scap']]
    v_lArm_lHnd = positions[linkName2idx['l_ufarm']] - positions[linkName2idx['l_larm']]
    #両足
    v_rThg_rShn = positions[linkName2idx['r_uleg']] - positions[linkName2idx['r_lglut']]
    v_rShn_rFot = positions[linkName2idx['r_talus']] - positions[linkName2idx['r_lleg']]
    v_rFot_rToe = positions[linkName2idx['r_foot_toe_sensor']] - \
                    positions[linkName2idx['r_foot_heel_sensor']]

    v_lThg_lShn = positions[linkName2idx['l_uleg']] - positions[linkName2idx['l_lglut']]
    v_lShn_lFot = positions[linkName2idx['l_talus']] - positions[linkName2idx['l_lleg']]
    v_lFot_lToe = positions[linkName2idx['l_foot_toe_sensor']] - \
                    positions[linkName2idx['l_foot_heel_sensor']]

    #体幹軸
    v_body = positions[linkName2idx['utorso']] - positions[linkName2idx['ltorso']]
    v_rvBody = positions[linkName2idx['ltorso']] - positions[linkName2idx['utorso']]
 
    #各向きのベクトルからクオータニオンを計算する
    def quat_from_two_vectors(a: glm.vec3, b: glm.vec3) -> glm.quat: #A to B(x,y,z,w)
        a = glm.normalize(a)
        b = glm.normalize(b)
        dot = glm.dot(a, b)
        if dot >= 1.0:
            return glm.quat(1, 0, 0, 0)  # 単位クオータニオン（回転なし）
        elif dot <= -1.0:
            # 真逆方向 → 任意の直交軸で180度回転
            orthogonal = glm.vec3(1, 0, 0) if abs(a.x) < 0.9 else glm.vec3(0, 1, 0)
            axis = glm.normalize(glm.cross(a, orthogonal))
            return glm.angleAxis(math.pi, axis)
        else:
            axis = glm.normalize(glm.cross(a, b))
            angle = math.acos(dot)
            return glm.angleAxis(angle, axis)
        
    quats = {}
    top_ax = glm.vec3(0, 0, 1)
    quats[quatName2idx['body']]    = quat_from_two_vectors(top_ax, v_body)
    quats[quatName2idx['rShldr']] = quat_from_two_vectors(v_rCol_rShl, v_rShl_rArm)
    quats[quatName2idx['rElbow']] = quat_from_two_vectors(v_rShl_rArm, v_rArm_rHnd)
    quats[quatName2idx['lShldr']] = quat_from_two_vectors(v_lCol_lShl, v_lShl_lArm)
    quats[quatName2idx['lElbow']] = quat_from_two_vectors(v_lShl_lArm, v_lArm_lHnd)
    quats[quatName2idx['rHip']]   = quat_from_two_vectors(v_rvBody, v_rThg_rShn)
    quats[quatName2idx['rKnee']]  = quat_from_two_vectors(v_rThg_rShn, v_rShn_rFot)
    quats[quatName2idx['rAnkle']] = quat_from_two_vectors(v_rShn_rFot, v_rFot_rToe)
    quats[quatName2idx['lHip']]   = quat_from_two_vectors(v_rvBody, v_lThg_lShn)
    quats[quatName2idx['lKnee']]  = quat_from_two_vectors(v_lThg_lShn, v_lShn_lFot)
    quats[quatName2idx['lAnkle']] = quat_from_two_vectors(v_lShn_lFot, v_lFot_lToe)
    return quats



def quaternion_viewer(motions:dict, body_occurrence_point:glm.vec3=glm.vec3(0,0,1)):
    '''
    クオータニオンの可視化をする
    主にデバッグ用
    各ステップごとに呼び出したmotion配列を引数に指定することで
    そのモーションが間違えてないか、人型のライン表示で確認可能
    '''

    v_body = body_occurrence_point #glm.vec3(0, 0, 1)
    v_b_neck = glm.vec3(0, 0, 0.5)
    q_b_neck = motions[quatName2idx['body']]
    v_b_neck = q_b_neck * v_b_neck


    v_n_lshd = glm.vec3(0, 0.2, 0)
    v_n_rshd = glm.vec3(0, -0.2, 0)

    #肩から肘
    q_lshd = motions[quatName2idx['lShldr']]
    q_rshd = motions[quatName2idx['rShldr']]
    v_lshd_lelb = q_lshd * (2.5 * v_n_lshd)
    v_rshd_relb = q_rshd * (2.5 * v_n_rshd)

    #肘から手
    q_lelb = motions[quatName2idx['lElbow']]
    q_relb = motions[quatName2idx['rElbow']]
    v_lelb_lhnd = q_lelb * v_lshd_lelb
    v_relb_rhnd = q_relb * v_rshd_relb

    #骨盤ベースから股関節
    v_b_lhip = glm.vec3(0, 0.2, 0)
    v_b_rhip = glm.vec3(0, -0.2, 0)

    #股関節から膝
    v_lhip_lkne = glm.vec3(0, 0, -0.5)
    v_rhip_rkne = glm.vec3(0, 0, -0.5)
    q_lhip = motions[quatName2idx['lHip']]
    q_rhip = motions[quatName2idx['rHip']]
    v_lhip_lkne = q_lhip * v_lhip_lkne
    v_rhip_rkne = q_rhip * v_rhip_rkne

    #膝から足首
    q_lKnee = motions[quatName2idx['lKnee']]
    q_rKnee = motions[quatName2idx['rKnee']]
    v_lkne_lank = q_lKnee * v_lhip_lkne
    v_rkne_rank = q_rKnee * v_rhip_rkne

    #足首からつま先
    q_lank = motions[quatName2idx['lAnkle']]
    q_rank = motions[quatName2idx['rAnkle']]
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




    


