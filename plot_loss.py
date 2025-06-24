import json
import matplotlib.pyplot as plt

# JSONデータ（必要に応じてファイルから読み込む場合は後述）
data = [
    {"epoch": 0, "train_loss": 7.932172710245306, "valid_loss": 6.854815165201823},
    {"epoch": 1, "train_loss": 6.877582333304665, "valid_loss": 5.661174456278483},
    {"epoch": 2, "train_loss": 5.810169284993952, "valid_loss": 4.642478942871094},
    {"epoch": 3, "train_loss": 4.925009835850108, "valid_loss": 3.8962340354919434},
    {"epoch": 4, "train_loss": 4.381194613196633, "valid_loss": 3.2364397843678794},
    {"epoch": 5, "train_loss": 3.853612021966414, "valid_loss": 2.7728464603424072},
    {"epoch": 6, "train_loss": 3.5099666010249746, "valid_loss": 2.5625462532043457},
    {"epoch": 7, "train_loss": 3.2702455412257803, "valid_loss": 2.2883957624435425},
    {"epoch": 8, "train_loss": 2.9113065979697486, "valid_loss": 1.9760093291600545},
    {"epoch": 9, "train_loss": 2.667582793669267, "valid_loss": 1.6831297874450684},
    {"epoch": 10, "train_loss": 2.503431114283475, "valid_loss": 1.6543755531311035},
    {"epoch": 11, "train_loss": 2.3149296749721873, "valid_loss": 1.2446399331092834},
    {"epoch": 12, "train_loss": 2.163297349756414, "valid_loss": 1.8600085179011028},
    {"epoch": 13, "train_loss": 2.5582353310151533, "valid_loss": 1.3991270065307617},
    {"epoch": 14, "train_loss": 1.9766984094272961, "valid_loss": 0.9597787658373514},
    {"epoch": 15, "train_loss": 1.7246528972278943, "valid_loss": 0.7878888050715128},
    {"epoch": 16, "train_loss": 1.5571726181290366, "valid_loss": 0.5975554883480072},
    {"epoch": 17, "train_loss": 1.5789604159918698, "valid_loss": 0.6903538306554159},
    {"epoch": 18, "train_loss": 1.5930341753092678, "valid_loss": 0.5717998544375101},
    {"epoch": 19, "train_loss": 1.3158380633050746, "valid_loss": 0.4228161970774333}
]

# epoch, train_loss, valid_loss を抽出
epochs = [d["epoch"] for d in data]
train_losses = [d["train_loss"] for d in data]
valid_losses = [d["valid_loss"] for d in data]

# グラフを描画
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label="Train Loss", marker='o')
plt.plot(epochs, valid_losses, label="Validation Loss", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
