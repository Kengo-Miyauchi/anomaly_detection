import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TorchDataset(Dataset):
    def __init__(self, x):
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        x = self.x[index]
        return x

def create_dataloader(X,batch_size,need_shuffle):
    dataloader = DataLoader(
        TorchDataset(X.astype(np.float32)),
        batch_size=batch_size,
        shuffle=need_shuffle,
        pin_memory=True,
        num_workers=4
    )
    return dataloader

def data_to_TabNetFeatures(exec_model,data,need_shuffle=False):
    dataloader = create_dataloader(data,exec_model.batch_size_tr,need_shuffle)
    # data to features as encoder output data
    features = []
    for batch_index, batch in enumerate(dataloader):
        # import pdb; pdb.set_trace()
        batch = batch.to(exec_model.device)
        try:
            step_outputs = exec_model.unsupervised_model.network.encoder(batch)[0]
        except Exception as e:
            print("\n"+f"Error occurred in batch {batch_index}: {e}")
            print(f"Batch shape: {batch.shape}")
            raise e
        encoder_out = sum(step for step in step_outputs)
        encoder_out = encoder_out.detach().cpu()
        del step_outputs, batch  # 中間データを削除
        torch.cuda.empty_cache()  # GPUメモリをクリア
        features.append(encoder_out)
    features = torch.cat(features)
    features = features.detach().cpu()
    return features


def data_to_framedFeatures(exec_model,data,num_frames,need_shuffle=False):
    dataloader = create_dataloader(data,exec_model.batch_size_tr,need_shuffle)
    # data to features as encoder output data
    framed_features = []
    for batch_index, batch in enumerate(dataloader):
        #import pdb; pdb.set_trace()
        batch = batch.to(exec_model.device)
        try:
            step_outputs = exec_model.unsupervised_model.network.encoder(batch)[0]
        except Exception as e:
            print("\n"+f"Error occurred in batch {batch_index}: {e}")
            print(f"Batch shape: {batch.shape}")
            raise e
        encoder_out = sum(step for step in step_outputs)
        encoder_out = encoder_out.detach().cpu()
        del step_outputs, batch
        i = 0
        while((len(encoder_out)-i)>=num_frames):
            tmp = []
            for j in range(num_frames):
                tmp.extend(encoder_out[j])
            i+=1
            framed_features.append(tmp)
            del tmp
        del encoder_out
        torch.cuda.empty_cache()
        #if (batch_index%100==0):print(f"{batch_index}/{len(dataloader)} batch:")
    return framed_features
