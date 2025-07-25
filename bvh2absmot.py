#BVHのモーション形式から主要関節のクオータニオン＋体幹の傾き3次元(中間表現)に変形する
#中間表現からアトラスのモーションに変形するアルゴリズムも必要
import math

import numpy as np
import glm
# from scipy.spatial.transform import Rotation as R

import bvhio

class bvh2motion():
    '''
    BVHファイルを関節角度21次元+体幹の傾き3次元に変形する
    腰3軸
    肩3軸 2
    肘1軸 2
    股関節3軸 2
    膝1軸 2
    足首2軸 2

    体幹の向き3次元

    初期化で開くファイルパスを指示
    get_motionで指定モーションの24次元ベクトルを返す
    '''

    joints_name = {'hip':0, 'abdomen':1, 'chest':2, 'neck':3, 'head':4, 
                    'rCollar':5, 'rShldr':6, 'rForeArm':7, 'rHand':8, 
                    'lCollar':9, 'lShldr':10, 'lForeArm':11, 'lHand':12, 
                    'rButtock':13, 'rThigh':14, 'rShin':15, 'rFoot':16, 
                    'lButtock':17, 'lThigh':18, 'lShin':19, 'lFoot':20, }
    
    quaternions_name = {'Body':0, 'rShldr':1, 'rElbow':2, 'lShldr':3, 'lElbow':4, 
                'rHip':5, 'rKnee':6, 'rAnkle':7, 'lHip':8, 'lKnee':9, 'lAnkle':10}

    def __init__(self, filepath):
        '''
        filepathのファイルを読み込んでhierarchy, rootとして読み込む
        '''
        self.filepath = filepath
        self.hierarchy = bvhio.readAsHierarchy(self.filepath)
        self.root = bvhio.readAsBvh(self.filepath)
        self.nFrame = self.root.FrameCount

  
    def convert_bvf_to_quaternion_motion(self, frame:int): # {glm.quat}: #A to B(x,y,z,w)
        '''
        frame目のモーションデータを、
        クオータニオンにすることで抽象化したモーションに変形する
        '''
        #frame番号のセット
        frame = max(0,min(self.nFrame, frame))
        hier.loadPose(frame)
        
        #計算に必要なジョイントの取得
        joints = [self.hierarchy.filter(name)[0] for name in self.joints_name.keys()]
        positions = [joint.PositionWorld for joint in joints]
        Ups = [joint.UpWorld for joint in joints]

        #計算に必要なベクトルの計算
        jn = self.joints_name
        #両腕
        v_rCol_rShl = positions[jn['rShldr']] - positions[jn['rCollar']]
        v_rShl_rArm = positions[jn['rForeArm']] - positions[jn['rShldr']]
        v_rArm_rHnd = positions[jn['rHand']] - positions[jn['rForeArm']]

        v_lCol_lShl = positions[jn['lShldr']] - positions[jn['lCollar']]
        v_lShl_lArm = positions[jn['lForeArm']] - positions[jn['lShldr']]
        v_lArm_lHnd = positions[jn['lHand']] - positions[jn['lForeArm']]

        #両足
        v_rThg_rShn = positions[jn['rShin']] - positions[jn['rThigh']]
        v_rShn_rFot = positions[jn['rFoot']] - positions[jn['rShin']]
        v_rFot_rToe = Ups[jn['rFoot']]

        v_lThg_lShn = positions[jn['lShin']] - positions[jn['lThigh']]
        v_lShn_lFot = positions[jn['lFoot']] - positions[jn['lShin']]
        v_lFot_lToe = Ups[jn['lFoot']]

        #体幹軸
        v_body = positions[jn['chest']] - positions[jn['hip']]
        v_rvBody = positions[jn['hip']] - positions[jn['abdomen']]

        #各座標のベクトルからクオータニオンを計算する
        
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
            
        quot = {}
        Y_ax = glm.vec3(0, 1, 0)

        quot['body']    = quat_from_two_vectors(Y_ax, v_body)
        quot['rShldr'] = quat_from_two_vectors(v_rCol_rShl, v_rShl_rArm)
        quot['rElbow'] = quat_from_two_vectors(v_rShl_rArm, v_rArm_rHnd)
        quot['lShldr'] = quat_from_two_vectors(v_lCol_lShl, v_lShl_lArm)
        quot['lElbow'] = quat_from_two_vectors(v_lShl_lArm, v_lArm_lHnd)
        quot['rHip']   = quat_from_two_vectors(v_rvBody, v_rThg_rShn)
        quot['rKnee']  = quat_from_two_vectors(v_rThg_rShn, v_rShn_rFot)
        quot['rAnkle'] = quat_from_two_vectors(v_rShn_rFot, v_rFot_rToe)
        quot['lHip']   = quat_from_two_vectors(v_rvBody, v_lThg_lShn)
        quot['lKnee']  = quat_from_two_vectors(v_lThg_lShn, v_lShn_lFot)
        quot['lAnkle'] = quat_from_two_vectors(v_lShn_lFot, v_lFot_lToe)
        return quot







joints_name = {'hip':0, 'abdomen':1, 'chest':2, 'neck':3, 'head':4, 
                    'rCollar':5, 'rShldr':6, 'rForeArm':7, 'rHand':8, 
                    'lCollar':9, 'lShldr':10, 'lForeArm':11, 'lHand':12, 
                    'rButtock':13, 'rThigh':14, 'rShin':15, 'rFoot':16, 
                    'lButtock':17, 'lThigh':18, 'lShin':19, 'lFoot':20, }

        

print('bvh to motion data')
hier = bvhio.readAsHierarchy('motions/02_01.bvh')
root = bvhio.readAsBvh('motions/02_01.bvh')
hier.printTree()
hier.loadPose(10)
print('0,0,0')
for i in range(10):
    print(f'0,{i},0')
for i in range(1):
    hier.loadPose(i)
    joints = [hier.filter(name)[0] for name in joints_name]
    pos = [j.PositionWorld for j in joints]
    for p in pos:
        print(f'{p[0]},{p[1]},{p[2]}')
        print(type(p))
    
readMotion = bvh2motion('motions/02_01.bvh')
dic = readMotion.convert_bvf_to_quaternion_motion(0)
for key, val in dic.items():
    print(f'{key}: {val}')

# for i in range(100):
#     hier.loadPose(i)

#     joint1 = hier.filter('rButtock')[0]
#     joint2 = hier.filter('rThigh')[0]
#     joint3 = hier.filter('rShin')[0]
#     joint4 = hier.filter('rFoot')[0]
#     # print(joint1.UpWorld)
#     # print(joint2.UpWorld)
#     # print(joint3.UpWorld)
#     # print(joint4.UpWorld)
#     x1 = joint1.UpWorld[0]*10
#     y1 = joint1.UpWorld[1]*10
#     z1 = joint1.UpWorld[2]*10
#     x2 = joint1.UpWorld[0]*10 + joint2.UpWorld[0]*10
#     y2 = joint1.UpWorld[1]*10 + joint2.UpWorld[1]*10
#     z2 = joint1.UpWorld[2]*10 + joint2.UpWorld[2]*10
#     x3 = joint1.UpWorld[0]*10 + joint2.UpWorld[0]*10 + joint3.UpWorld[0]*10
#     y3 = joint1.UpWorld[1]*10 + joint2.UpWorld[1]*10 + joint3.UpWorld[1]*10
#     z3 = joint1.UpWorld[2]*10 + joint2.UpWorld[2]*10 + joint3.UpWorld[2]*10
#     x4 = joint1.UpWorld[0]*10 + joint2.UpWorld[0]*10 + joint3.UpWorld[0]*10 + joint4.UpWorld[0]*10
#     y4 = joint1.UpWorld[1]*10 + joint2.UpWorld[1]*10 + joint3.UpWorld[1]*10 + joint4.UpWorld[1]*10
#     z4 = joint1.UpWorld[2]*10 + joint2.UpWorld[2]*10 + joint3.UpWorld[2]*10 + joint4.UpWorld[2]*10
#     print(f'{x1},{y1},{z1}')
#     print(f'{x2},{y2},{z2}')
#     print(f'{x3},{y3},{z3}')
#     print(f'{x4},{y4},{z4}')
    # print('****')
    
    # joint1 = hier.filter('lButtock')[0]
    # joint2 = hier.filter('lThigh')[0]
    # joint3 = hier.filter('lShin')[0]
    # joint4 = hier.filter('lFoot')[0]
    # # print(joint1.UpWorld)
    # # print(joint2.UpWorld)
    # # print(joint3.UpWorld)
    # # print(joint4.UpWorld)
    # x1 = joint1.PositionWorld[0]*10
    # y1 = joint1.PositionWorld[1]*10
    # z1 = joint1.PositionWorld[2]*10
    # x2 = joint1.PositionWorld[0]*10 + joint2.PositionWorld[0]*10
    # y2 = joint1.PositionWorld[1]*10 + joint2.PositionWorld[1]*10
    # z2 = joint1.PositionWorld[2]*10 + joint2.PositionWorld[2]*10
    # x3 = joint1.PositionWorld[0]*10 + joint2.PositionWorld[0]*10 + joint3.PositionWorld[0]*10
    # y3 = joint1.PositionWorld[1]*10 + joint2.PositionWorld[1]*10 + joint3.PositionWorld[1]*10
    # z3 = joint1.PositionWorld[2]*10 + joint2.PositionWorld[2]*10 + joint3.PositionWorld[2]*10
    # x4 = joint1.PositionWorld[0]*10 + joint2.PositionWorld[0]*10 + joint3.PositionWorld[0]*10 + joint4.PositionWorld[0]*10
    # y4 = joint1.PositionWorld[1]*10 + joint2.PositionWorld[1]*10 + joint3.PositionWorld[1]*10 + joint4.PositionWorld[1]*10
    # z4 = joint1.PositionWorld[2]*10 + joint2.PositionWorld[2]*10 + joint3.PositionWorld[2]*10 + joint4.PositionWorld[2]*10
    # print(f'{x1},{y1},{z1}')
    # print(f'{x2},{y2},{z2}')
    # print(f'{x3},{y3},{z3}')
    # print(f'{x4},{y4},{z4}')
    
    # print(f'{joint1.UpWorld[0]*10:.4f},{joint1.UpWorld[1]*10:.4f},{joint1.UpWorld[2]*10:.4f} ')
    # print(f'{joint1.UpWorld[0]*10:.4f},{joint1.UpWorld[1]*10:.4f},{joint1.UpWorld[2]*10:.4f} ')

# for joint, index, depth in hier.layout():
#     print(f'{joint.PositionWorld} {joint.UpWorld} {joint.Name} {index} {depth}')
#     print(joint.Children)
