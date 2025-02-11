import torch
import pandas as pd
import numpy as np
import os
import gc
from matplotlib import pyplot as plt

path_to_pretrained = 'model/haenkaze/tabnet-pretrain-40dim2'
unsupervised_model=torch.load(path_to_pretrained+'/pretrained.pth')
start = 1

# 学習履歴から再構成ロスと検証セットのロスを取得
train_loss = unsupervised_model.history['loss']
valid_loss = unsupervised_model.history['val_0_unsup_loss_numpy']
train_loss = train_loss[start-1:]
valid_loss = valid_loss[start-1:]

# グラフの作成
plt.figure(figsize=(10, 6))
plt.plot(range(start, start+len(train_loss)), train_loss, label="Train Loss")
plt.plot(range(start, start+len(valid_loss)), valid_loss, label="Validation Loss")
plt.title("Reconstruction Loss during Pretraining", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
img_path = path_to_pretrained+"/train_curve_"+str(start)+"-"+str(start+len(train_loss)-1)+".png"
plt.savefig(img_path)
print(img_path)
print(f"min train loss: {min(train_loss)}")
print(f"min valid loss: {min(valid_loss)}")
print(f"final train loss: {train_loss[-1]}")
print(f"final valid loss: {valid_loss[-1]}")