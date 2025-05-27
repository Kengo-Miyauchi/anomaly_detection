from matplotlib import pyplot as plt
N=[0,1,2,3,4,5,6,7]
frequency_list =['1S','10S_sampled','10S','1M','10M']
scores = [0.78,0.78,0.77,0.74,0.77,0.73,0.51,0.41]
metric = "recall"
model = "tabnet-gmm"
plt.figure(figsize=(10, 6))
plt.plot(range(len(N)), scores)
plt.title(f"{metric} curve", fontsize=14)
plt.xlabel("N", fontsize=12)
plt.ylabel(f"{metric}", fontsize=12)
plt.legend()
img_path = f"result/haenkaze/{model}/{metric}_curve.png"
plt.savefig(img_path)
print(f"{metric}: {img_path}")