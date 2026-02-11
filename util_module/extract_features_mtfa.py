import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TorchDataset(Dataset):
    def __init__(self, x):
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index]

@torch.no_grad()
def data_to_MTFAFeatures(exec_model, X_windows_3d: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    exec_model: ExecModelMTFA
    X_windows_3d: [N, L, C]
    return: [N, 40]
    """
    exec_model.model.eval()

    loader = DataLoader(
        TorchDataset(X_windows_3d.astype(np.float32)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    feats = []
    for x in loader:
        x = x.to(exec_model.device)  # [B, L, C]

        tematt, freatt = exec_model.model(x)

        # tematt/freatt は list
        # 最後に append(rec) しているので rec は [B, T, D]
        rec_tem = tematt[-1]  # [B, L, D]
        rec_fre = freatt[-1]  # [B, L, D]

        # window内pooling → [B, D]
        z_tem = rec_tem.mean(dim=1)
        z_fre = rec_fre.mean(dim=1)

        # concat → [B, 2D]
        z = torch.cat([z_tem, z_fre], dim=-1)

        # 射影 → [B, 40]
        z40 = exec_model.proj(z)

        feats.append(z40.detach().cpu().numpy())

    return np.concatenate(feats, axis=0).astype(np.float32)