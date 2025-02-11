import numpy as np
from pytorch_tabnet.pretraining import TabNetPretrainer

def frame_data(features,num_frames):
    framed = []
    for i in range(len(features)):
        tmp = []
        if (len(features)-i)>=num_frames:
            for j in range(num_frames):
                tmp.extend(features[i+j])
            framed.append(tmp)
    del features
    return framed


# サンプルデータ
num_samples, num_features = 100, 20
data = np.random.rand(num_samples, num_features)

# フレーム化
frame_size = 5
def create_frames(data, frame_size):
    num_samples, num_features = data.shape
    num_frames = num_samples - frame_size + 1
    frames = np.array([data[i:i + frame_size] for i in range(num_frames)])
    return frames

framed_data = create_frames(data, frame_size)

# フレームデータを2次元に変換
num_frames, frame_size, num_features = framed_data.shape
X_train = framed_data.reshape(num_frames, frame_size * num_features)
print(X_train.shape)

# TabNetPretrainerの定義
pretrainer = TabNetPretrainer(
    input_dim=frame_size * num_features,  # フレームごとの特徴量数
    device_name='cuda'  # GPUを使用
)

# 事前学習
pretrainer.fit(
    X_train=X_train,
    eval_set=[(X_train, X_train)],  # 検証データ
    pretraining_ratio=0.2,
    max_epochs=10,
    patience=5,
    batch_size=256,
    virtual_batch_size=128
)
