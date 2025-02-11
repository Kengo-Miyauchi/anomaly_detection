from matplotlib import pyplot as plt
import pandas as pd

frequency_list =['1S','10S','10S_sampled','1M','10M']
model = "tabnet-gmm"
df = pd.read_csv(f"result/haenkaze/{model}/scores.csv")
#import pdb; pdb.set_trace()
N=[0,1,2,3,4,5,6,7]
metric = "precision"
plt.figure(figsize=(10, 6))
for i in range(0,len(frequency_list)):
    precisions = []
    for j in range(len(N)):
        precisions.append(df[f"p{j}"][i])
    plt.plot(range(len(N)), precisions, label=frequency_list[i])
plt.title(f"{metric} Curve", fontsize=14)
plt.xlabel("N", fontsize=12)
plt.ylabel(f"{metric}", fontsize=12)
plt.legend()
img_path = f"result/haenkaze/{model}/{metric}_curve.png"
plt.savefig(img_path)
print(f"{metric}: {img_path}")

metric = "recall"
plt.figure(figsize=(10, 6))
for i in range(0,len(frequency_list)):
    recalls = []
    for j in range(len(N)):
        recalls.append(df[f"r{j}"][i])
    plt.plot(range(len(N)), recalls, label=frequency_list[i])
plt.xlabel("N", fontsize=12)
plt.ylabel("Recall", fontsize=12)
plt.legend()
img_path = f"result/haenkaze/{model}/{metric}_curve.png"
plt.savefig(img_path)
print(f"{metric}: {img_path}")

metric = "F1"
plt.figure(figsize=(10, 6))
for i in range(0,len(frequency_list)):
    f1_scores = []
    for j in range(len(N)):
        f1_scores.append(df[f"f{j}"][i])
    plt.plot(range(len(N)), f1_scores, label=frequency_list[i])
plt.xlabel("N", fontsize=12)
plt.ylabel("F1", fontsize=12)
plt.legend()
img_path = f"result/haenkaze/{model}/{metric}_curve.png"
plt.savefig(img_path)
print(f"{metric}: {img_path}")