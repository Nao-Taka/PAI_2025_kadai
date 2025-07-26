import numpy as np
import pybullet as p


Flag_CONTACT = {
"contactFlag": 0,
"bodyUniqueIdA": 1,
"bodyUniqueIdB": 2,
"linkIndexA": 3,
"linkIndexB": 4,
"positionOnA": 5,
"positionOnB": 6,
"contactNormalOnB": 7,
"contactDistance": 8,
"normalForce": 9,
"lateralFriction1": 10,
"lateralFrictionDir1": 11,
"lateralFriction2": 12,
"lateralFrictionDir2": 13
}

def sensor_debug(model, floor):
    '''
    デバッグ用
    モデルと床との接触をチェック
    接触している場合、法線ベクトルと接触点を表示する
    '''
    contacts = p.getContactPoints(model, floor)
    print(f'n contact {len(contacts)}')
    # print(len(contacts))
    for cat in contacts:
        print(cat)
        dist = cat[Flag_CONTACT['contactDistance']]
        if dist < 0.001:
            start = np.array(cat[Flag_CONTACT['positionOnA']])
            normal = np.array(cat[Flag_CONTACT['contactNormalOnB']])
            force  = cat[Flag_CONTACT['normalForce']]
            #法線ベクトル　×　力の大きさを参考に接触点から伸びる直線を引いて視覚化する
            p.addUserDebugLine(start,
                               start + normal * force/100,
                               lineColorRGB=[1., 0, 1.])
                            #    lifeTime=0.1)
            p.addUserDebugPoints([start],
                                 pointColorsRGB=[[0., 1., 0.]],
                                 pointSize=10)
                                #  lifeTime=0.1)


