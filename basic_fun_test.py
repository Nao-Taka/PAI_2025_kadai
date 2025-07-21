#PytorchとPybulletの動作確認

import torch

# Tensorを作成して、GPUが使えるか確認
x = torch.rand(3, 3)
print("Tensor:")
print(x)

if torch.cuda.is_available():
    print("✅ CUDA (GPU) is available!")
else:
    print("❌ CUDA (GPU) is NOT available.")


import pybullet as p
import pybullet_data
import time

# 物理シミュレーションのGUIモードで接続
physicsClient = p.connect(p.GUI)  # または p.DIRECT で非表示モード

# 環境初期化
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# プレーンとロボット（例：r2d2）を読み込む
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 1])

# 数秒間表示（10秒）
for _ in range(240):
    p.stepSimulation()
    time.sleep(1.0 / 24.0)

p.disconnect()
