import torch
import math
import pickle
import json
import os
import csv
import argparse
from tqdm import tqdm
import random
import pandas as pd
from util_module.extract_features import create_dataloader

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parse(data, part, encoder_model, out_dpath, scada_dpath):
    all_embeddings = []
    all_captions = []
    
    for i, item in enumerate(tqdm(data)):
        file_id = item[0]
        caption = item[1]
        #time_range = item[2]
        # SCADAファイル読み込み（./scada/001.csv）
        scada_path = os.path.join(scada_dpath, f"{file_id}.csv")
        with open(scada_path, "r") as f:
            df = pd.read_csv(scada_path, skiprows=1)  # ヘッダー行をスキップ
            series = df.values
        
        embedding_chunks = []

        for j in range(6):
            start_index = j * (series.shape[0] // 6)
            end_index = (j + 1) * (series.shape[0] // 6)
            chunk_series = series[start_index:end_index]

            dataloader = create_dataloader(chunk_series, 32, False)
            features = []

            for batch_index, batch in enumerate(dataloader):
                batch = batch.to(DEVICE)
                with torch.no_grad():
                    step_outputs, _ = encoder_model(batch)
                encoder_out = sum(step for step in step_outputs)
                encoder_out = encoder_out.detach().cpu()
                features.append(encoder_out)
                del step_outputs, batch
                torch.cuda.empty_cache()

            features = torch.cat(features, dim=0)        # [T, 40]
            chunk_embedding = features.mean(dim=0)       # [40]
            embedding_chunks.append(chunk_embedding)     # List[6 x 40]
            #import pdb; pdb.set_trace()

        # ここで連結 → [240]
        final_embedding = torch.cat(embedding_chunks).unsqueeze(0)  # [1, 240]
        all_embeddings.append(final_embedding)


        all_captions.append({
            "scada_embedding": i,
            "caption": caption,
            #"time_range": time_range,
        })

    out_data = {
        "scada_embedding": torch.cat(all_embeddings, dim=0),
        "captions": all_captions
    }

    os.makedirs(out_dpath, exist_ok=True)
    data_fpath = os.path.join(out_dpath, f"{part}.pkl")
    pickle.dump(out_data, open(data_fpath, "wb"))
    print(f"Saved {part} data to {data_fpath}.")
    return data_fpath

def prepare_data(captions_fpath, encoder_model, test_ratio, valid_ratio, train_ratio, shuffle=False, scada_dpath=None):
    assert sum([test_ratio, valid_ratio, train_ratio]) <= 1.0

    all_data = list(csv.reader(open(captions_fpath)))
    del all_data[0]  # ヘッダー行を削除
    
    if shuffle:
        random.shuffle(all_data)

    total = len(all_data)
    test_data = all_data[:int(total * test_ratio)]
    valid_data = all_data[int(total * test_ratio):int(total * (test_ratio + valid_ratio))]
    train_data = all_data[int(total * (test_ratio + valid_ratio)):int(total * (test_ratio + valid_ratio + train_ratio))]

    out_dpath = os.path.join(os.path.dirname(captions_fpath), "processed-scada")
    os.makedirs(out_dpath, exist_ok=True)

    test_data_fpath = parse(test_data, "test", encoder_model, out_dpath, scada_dpath)
    valid_data_fpath = parse(valid_data, "valid", encoder_model, out_dpath, scada_dpath)
    train_data_fpath = parse(train_data, "train", encoder_model, out_dpath, scada_dpath)

    return test_data_fpath, valid_data_fpath, train_data_fpath

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_fpath", type=str, required=True, help="CSV path with SCADA file IDs and captions")
    parser.add_argument("--encoder_model_path", type=str, required=True, help="Path to pretrained TabNet model directory")
    parser.add_argument("--scada_dpath", type=str, required=True, help="Directory containing SCADA CSV files")
    args = parser.parse_args()

    # モデルの読み込み
    path_to_pretrained = args.encoder_model_path
    unsupervised_model = (torch.load(os.path.join(path_to_pretrained, "pretrained.pth"), weights_only=False))
    encoder = unsupervised_model.network.encoder

    # 実行
    prepare_data(
        captions_fpath=args.captions_fpath,
        encoder_model=encoder,
        test_ratio=0.1,
        valid_ratio=0.1,
        train_ratio=0.8,
        shuffle=True,
        scada_dpath=args.scada_dpath
    )
