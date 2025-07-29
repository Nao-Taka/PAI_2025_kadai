import numpy as np
import cv2
import pybullet as p


#補助的な作業を行うクラスなどを保存

#画面録画をする
class recode():
    def __init__(self, filename, width=640, height=480, fps=24):
        #Pybulletの出力をそのまま入力すること1
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


class saveMovie():
    def __init__(self, filename, fps=24):
        '''
        Opencvに合わせて成型された動画データをフレームごとに保存していく
        '''
        self.filename = filename
        self.fps = fps
        self.width = None
        self.height = None
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = None
        
    def capture(self, img_arr):
        if self.video is None:
            self.height, self.width, _ = img_arr.shape
            self.video = cv2.VideoWriter(self.filename + '.mp4', self.fourcc, 
                                         self.fps, (self.width, self.height))

        self.video.write(img_arr)
    
    def __del__(self):
        self.video.release()
