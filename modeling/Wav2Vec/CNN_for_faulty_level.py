import torch.nn as nn
import fairseq.fairseq.checkpoint_utils


class Wav2VecCNNForFaultyLevel(nn.Module):
    def __init__(self, wav2vec_path, bottleneck, output):
        super().__init__()

        self.wav2vec, _, _ = fairseq.fairseq.checkpoint_utils.load_model_ensemble_and_task([wav2vec_path])
        self.wav2vec = self.wav2vec[0]

        self.CNN = nn.Sequential(
            # (N, 1, 512, X)
            nn.Conv2d(1, 64, 3, 2),
            nn.ReLU(True),
            # (N, 128, 256, X)
            nn.Conv2d(64, 128, 3, 2),
            nn.ReLU(True),
            # (N, 256, 128, X)
            nn.Conv2d(128, 256, 2, 1),
            nn.ReLU(True),
            # (N, 512, 64, X)
            nn.Conv2d(256, 256, 2, 1),
            nn.ReLU(True),
            # (N, 512, 32, X)
            # nn.Conv2d(256, 256, 2, 1),
            # nn.ReLU(True),
            # (N, 512, 32, X)
            nn.Conv2d(256, bottleneck, 2, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            # (N, bottleneck, 1, 1)
        )

        self.estimator = nn.Sequential(
            nn.Linear(bottleneck, output),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        bn = self.wav2vec.feature_extractor(x)
        bn = bn.reshape(bn.shape[0], 1, bn.shape[1], bn.shape[2])
        bn = self.CNN(bn)
        bn = bn.reshape(bn.shape[0], -1)
        output = self.estimator(bn)
        return output, bn


class Wav2VecCNNForFaultyLevel2(nn.Module):
    def __init__(self, wav2vec_path, bottleneck, output):
        super().__init__()

        self.wav2vec, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([wav2vec_path])
        self.wav2vec = self.wav2vec[0]

        self.CNN = nn.Sequential(
            # (N, 1, 512, X)
            nn.Conv2d(1, bottleneck, 1, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            # (N, bottleneck, 1, 1)
        )

        self.estimator = nn.Sequential(
            nn.Linear(bottleneck, output),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        bn = self.wav2vec.feature_extractor(x)
        bn = bn.reshape(bn.shape[0], 1, bn.shape[1], bn.shape[2])
        bn = self.CNN(bn)
        bn = bn.reshape(bn.shape[0], -1)
        output = self.estimator(bn)
        return output, bn

