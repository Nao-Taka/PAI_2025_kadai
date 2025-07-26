import numpy as np
import cv2
import pybullet as p


#補助的な作業を行うクラスなどを保存

#画面録画をする
class recode():
    def __init__(self, filename, width=640, height=480, fps=24):
        #最低限filenameを設定して準備、ファイル名に拡張子は含めないこと
        #相対パスは可能
        self.filename = filename
        self.width = width
        self.height = height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = cv2.VideoWriter(filename + '.mp4', fourcc, fps, (width, height))

    def capture(self):
        img_arr = p.getCameraImage(self.width, self.height)
        rgb = np.reshape(img_arr[2], (self.height, self.width, 4))[:,:,:3]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.video.write(rgb)
    
    def __del__(self):
        self.video.release()