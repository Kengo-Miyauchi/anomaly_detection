import os
import yaml
import torch
import torch.nn as nn
import logging
import numpy as np
from matplotlib import pyplot as plt

from util_module.TFMAE.model.MTFAE import MTFA  # パスは環境に合わせて調整


def my_kl_loss(p, q):
    # solver.py と同じ
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)


class ExecModelTFMAE:
    """
    TabNet版ExecModelの流儀に合わせた MTFA(=TFMAE repo) 版ExecModel
    - yaml config 読み込み
    - out_dir / path_to_pretrained 自動生成
    - pretrained.pt があればロード、なければ学習して保存
    - solver.py と同じ loss (= con_loss - adv_loss) で学習
    - train_curve() で loss 推移を保存
    """

    def __init__(
        self,
        device,
        config,
        dataset_name,
        model_name,
        train_loader=None,
        valid_loader=None,
        refit=False,
        proj_dim=40,
        optimizer_params=None,
        max_epochs=None,
        # MTFA hyperparams
        c_in=None,
        d_model=512,
        e_layers=3,
        fr=0.4,
        tr=0.5,
        seq_size=None,
        sequence_length=None,  # win_size
        covariance_type="diag",
        path_to_pretrained=None,
    ):
        self.device = device
        self.config = config
        self.dataset_name = dataset_name
        self.model_name = model_name

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.refit = refit

        self.proj_dim = proj_dim
        self.optimizer_params = optimizer_params or {"lr": 1e-3}
        self.max_epochs = max_epochs or 50

        # MTFA params
        self.c_in = c_in
        self.d_model = d_model
        self.e_layers = e_layers
        self.fr = fr
        self.tr = tr
        self.sequence_length = sequence_length  # win_size
        self.seq_size = seq_size if seq_size is not None else sequence_length
        self.covariance_type = covariance_type

        # config読込（TabNet版踏襲）
        self.set_params_from_file(config)
        self.conditions = self.gather_conditions()

        # out_dir
        self.out_dir = self.set_out_dir()
        os.makedirs(self.out_dir, exist_ok=True)

        # 保存先
        if path_to_pretrained is None:
            self.path_to_pretrained = f"/mnt/iot-qnap5/miyauchi/model/{dataset_name}/{model_name}-{self.proj_dim}dim"
            if self.sequence_length is not None:
                self.path_to_pretrained += f"-{self.sequence_length}seq"
        else:
            self.path_to_pretrained = path_to_pretrained
        os.makedirs(self.path_to_pretrained, exist_ok=True)

        self.ckpt_path = os.path.join(self.path_to_pretrained, "pretrained.pt")

        # ロードor学習
        if os.path.exists(self.ckpt_path):
            print(f"Load model from {self.ckpt_path}")
            self.set_log(filepath=os.path.join(self.path_to_pretrained, "pretraining_refit.log" if self.refit else "pretraining.log"))

            self.set_unsupervised_model()
            self.load_ckpt(self.ckpt_path)

            if self.refit:
                logging.info("Starting MTFA pretraining refit...")
                self.fit_model()
                logging.info("MTFA pretraining refit finished.")
                self.save_ckpt(os.path.join(self.path_to_pretrained, "pretrained_refit.pt"))
        else:
            self.set_log(filepath=os.path.join(self.path_to_pretrained, "pretraining.log"))

            self.set_unsupervised_model()
            logging.info("Starting MTFA pretraining...")
            self.fit_model()
            logging.info("MTFA pretraining finished.")
            self.save_ckpt(self.ckpt_path)

    # ---- モデル生成（TabNetのset_unsupervised_model相当）----
    def set_unsupervised_model(self):
        if self.c_in is None:
            raise ValueError("c_in is None. Set c_in (num features).")
        if self.sequence_length is None:
            raise ValueError("sequence_length is None. Set sequence_length (win_size).")
        if self.seq_size is None:
            raise ValueError("seq_size is None. Set seq_size (moving window in TemEnc).")

        # MTFA本体（solver.py と同じ引数）
        self.model = MTFA(
            win_size=self.sequence_length,
            seq_size=self.seq_size,
            c_in=self.c_in,
            c_out=self.c_in,
            d_model=self.d_model,
            e_layers=self.e_layers,
            fr=self.fr,
            tr=self.tr,
            dev=self.device,
        ).to(self.device)

        # GMM用の40次元射影（※学習には使わないが、保存しておくと推論側で統一できる）
        self.proj = nn.Linear(2 * self.d_model, self.proj_dim).to(self.device)

        return self.model

    # ---- solver.py と同じ loss（con_loss - adv_loss）----
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        tematt, freatt = self.model(x)

        adv_loss = 0.0
        con_loss = 0.0

        for u in range(len(freatt)):
            # freatt[u] / sum の正規化（solver.pyと同じ）
            fre_u = freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)

            adv_loss = adv_loss + (
                torch.mean(my_kl_loss(tematt[u], fre_u.detach())) +
                torch.mean(my_kl_loss(fre_u.detach(), tematt[u]))
            )

            con_loss = con_loss + (
                torch.mean(my_kl_loss(fre_u, tematt[u].detach())) +
                torch.mean(my_kl_loss(tematt[u].detach(), fre_u))
            )

        adv_loss = adv_loss / len(freatt)
        con_loss = con_loss / len(freatt)

        loss = con_loss - adv_loss
        return loss

    # ---- 学習ループ（TabNetのfit_model相当）----
    def fit_model(self):
        if self.train_loader is None:
            raise ValueError("train_loader is None. Provide DataLoader for training.")

        self.train_loss_list = []
        self.val_loss_list = []

        # solver.py は model.parameters() のみ最適化
        # proj は特徴抽出用なので、ここでは optimizer に入れない（入れてもlossがprojに依存しないので意味がない）
        optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_params)

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()

            total = 0.0
            n = 0
            for x in self.train_loader:
                x = x.float().to(self.device)

                loss = self.compute_loss(x)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total += float(loss.item())
                n += 1

            train_loss = total / max(n, 1)
            self.train_loss_list.append(train_loss)

            msg = f"Epoch: {epoch} | Train Loss: {train_loss:.7f}"

            # validation
            if self.valid_loader is not None:
                self.model.eval()
                vlist = []
                with torch.no_grad():
                    for x in self.valid_loader:
                        x = x.float().to(self.device)
                        vloss = self.compute_loss(x)
                        vlist.append(float(vloss.item()))
                vali_loss = float(np.mean(vlist)) if len(vlist) else float("nan")
                self.val_loss_list.append(vali_loss)
                msg += f"  Vali Loss: {vali_loss:.7f}"

            logging.info(msg)
            print(msg)

    # ---- 保存/ロード ----
    def save_ckpt(self, path):
        payload = {
            "model_state": self.model.state_dict(),
            "proj_state": self.proj.state_dict(),
            "conditions": self.conditions,
            "config": self.config,
        }
        torch.save(payload, path)
        print(f"Saved: {path}")

    def load_ckpt(self, path):
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state"], strict=True)

        # 古いckptにprojが無い可能性も考慮
        if "proj_state" in payload and hasattr(self, "proj"):
            self.proj.load_state_dict(payload["proj_state"], strict=True)

    # ---- ログ設定（basicConfigの弱点回避）----
    def set_log(self, filepath):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        fh = logging.FileHandler(filepath, mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        print(f"log file : {filepath}")

    # ---- 出力ディレクトリ（TabNet版の流儀）----
    def set_out_dir(self):
        out_dir = f"result/{self.dataset_name}/{self.model_name}/{self.proj_dim}dim"
        if self.covariance_type is not None and self.covariance_type != "full":
            out_dir = out_dir + f"-{self.covariance_type}"
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    # ---- config読込（未知キーはwarnで無視）----
    def set_params_from_dict(self, params):
        for key, value in (params or {}).items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"[Warn] Unknown parameter ignored: {key}")
        return self

    def set_params_from_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                params = yaml.safe_load(f) or {}
                self.set_params_from_dict(params)
        except FileNotFoundError:
            print(f"Error: Config file not found at {file_path}")
        except yaml.YAMLError as e:
            print(f"Error: Failed to parse YAML file {file_path}: {e}")
        return self

    def gather_conditions(self):
        return [
            ("proj_dim", self.proj_dim),
            ("optimizer_params", self.optimizer_params),
            ("max_epochs", self.max_epochs),
            ("sequence_length", self.sequence_length),
            ("seq_size", self.seq_size),
            ("c_in", self.c_in),
            ("d_model", self.d_model),
            ("e_layers", self.e_layers),
            ("fr", self.fr),
            ("tr", self.tr),
            ("covariance_type", self.covariance_type),
        ]

    # ---- 学習曲線（TabNet版のtrain_curve踏襲）----
    def train_curve(self, start=1):
        if not hasattr(self, "train_loss_list"):
            print("No training history. Run fit_model() first.")
            return

        start = max(1, start)
        train_loss = self.train_loss_list[start-1:]
        val_loss = self.val_loss_list[start-1:] if hasattr(self, "val_loss_list") and len(self.val_loss_list) > 0 else None

        plt.figure(figsize=(10, 6))
        plt.plot(range(start, start + len(train_loss)), train_loss, label="Train Loss")
        if val_loss is not None:
            plt.plot(range(start, start + len(val_loss)), val_loss, label="Validation Loss")
        plt.title("Loss during Pretraining (MTFA)", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend()

        img_path = os.path.join(self.path_to_pretrained, f"train_curve_{start}-{start+len(train_loss)-1}.png")
        plt.savefig(img_path)
        print(f"Train curve: {img_path}")
