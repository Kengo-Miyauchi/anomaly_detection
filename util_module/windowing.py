import numpy as np

def make_windows_3d(X2d: np.ndarray, L: int, stride: int = 1) -> np.ndarray:
    """
    X2d: [T, C] -> [N, L, C]
    """
    X2d = np.asarray(X2d)
    T, C = X2d.shape
    if T < L:
        return np.empty((0, L, C), dtype=np.float32)

    idx = range(0, T - L + 1, stride)
    return np.stack([X2d[i:i+L] for i in idx]).astype(np.float32)

def window_timestamps(ts, L: int, stride: int = 1, mode: str = "median"):
    """
    ts: 長さTのtimestamp配列
    return: 長さNの代表timestamp
    mode:
      - "median": window内の中央値（あなたのmedian_timestamps相当）
      - "center": window中央
    """
    ts = np.asarray(ts)
    T = len(ts)
    if T < L:
        return np.array([])

    reps = []
    for i in range(0, T - L + 1, stride):
        w = ts[i:i+L]
        if mode == "center":
            reps.append(w[L // 2])
        else:
            reps.append(np.median(w))
    return np.asarray(reps)

def fit_standardizer(X2d: np.ndarray, eps: float = 1e-6):
    """
    train(正常)からmean/stdを作る
    """
    X2d = np.asarray(X2d, dtype=np.float32)
    mu = X2d.mean(axis=0, keepdims=True)
    std = X2d.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return mu, std

def apply_standardizer(X2d: np.ndarray, mu, std):
    X2d = np.asarray(X2d, dtype=np.float32)
    return (X2d - mu) / std