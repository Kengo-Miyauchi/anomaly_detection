import argparse
import sys
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from collections import OrderedDict
from torchinfo import summary
from multiprocessing import Pool
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(".")
from modeling.mobilenet.mobilenetv2_IDdiscriminator import MelMobileNetV2_IDdiscriminator10_1
from modeling.Flow_based.RealNVP import RealNVP
from modeling.Flow_based.FlowPP import FlowPP
from modeling.Flow_based.ResFlow import ResFlow
from modeling.pytorch.early_stopping import EarlyStopping
from util.tools import pathlib_mkdir
from util.tools import load_list_file
from util.tools import read_length_shift
from util.manage_list import unroll
from scoring.score import ScoreSave
from scoring.f_value import f_value
from scoring.overdetection_accuracy import calc_healthy_sample_accuracy
from scoring.overdetection_accuracy import calc_anomaly_sample_accuracy

register_matplotlib_converters()

####################
# fix pytorch seed
####################
# torch.manual_seed(0)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# np.random.seed(0)

###################
# argparse
###################
parser = argparse.ArgumentParser()
parser.add_argument("target", type=str)
parser.add_argument("domain", type=str)
parser.add_argument("--part", default="MainBearing1", type=str)
parser.add_argument("--files", default=100, type=int)

parser.add_argument("--window-shift", dest="w_and_s", default="w5.12_s0.1")
parser.add_argument("--splitsize", default=8, type=int)

parser.add_argument("--mid1", default=8, type=int)
parser.add_argument("--bottleneck", default=16, type=int)
parser.add_argument("--batchsize", default=128, type=int)
parser.add_argument("--epoch", default=20, type=int)
parser.add_argument("--valid", action="store_true")
parser.add_argument("--nf", "--force-nn-training", dest="network", action="store_true")

parser.add_argument("--nf_type", default="ResFlow", type=str)
parser.add_argument("--nf_layers", default=32, type=int)
parser.add_argument("--nf_epoch", default=300, type=int)
parser.add_argument("--percentile", type=int, dest="designed_percentile", default=99)
parser.add_argument("-f", "--force_training", dest="force", action="store_true")

parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
parser.add_argument("-t", "--tmp", dest="tmp", action="store_true")
parser.add_argument("-l", "--log", dest="log", action="store_true")
parser.add_argument("--cuda", type=str, default="cuda:0")

parser.add_argument("--core", default=6, type=int)
parser.add_argument("--visual", action="store_true")
parser.add_argument("--noinfo", action="store_true")

args = parser.parse_args()
params = vars(args)

##############################
# read json file
##############################
config_file = Path("config.yml")
with config_file.open(mode="r") as f:
    config = yaml.safe_load(f)

#################
# setting
#################
# input parameter
domain = args.domain
w_and_s = args.w_and_s
w_length, w_shift = read_length_shift(w_and_s)
sampling_freq = 25600

# WindTurbine parameter
target = args.target
part = args.part

# feature extractor parameter
width = args.splitsize
mid1 = args.mid1
bottleneck = args.bottleneck
batchsize = args.batchsize
n_epoch = args.epoch

architecture = f"mobilenet10_1_width{width}_mid1{mid1}_bn{bottleneck}"

# collator parameter
files = args.files
nf_layers = args.nf_layers
nf_epoch = args.nf_epoch

# other parameter
n_core = args.core
output_log = args.verbose

###########################
# folder and file location
###########################
cms_dir = Path(config["cms_dir"])
target_list_dir = cms_dir / "list" / "iwaya07_mutsu-ogawara21"
all_list_dir = cms_dir / "list/normal.MainBearing1"
feature_dir = cms_dir / "feature" / domain / w_and_s
vib_dir1 = cms_dir / "vibration"
vib_dir2 = Path(config["vib_dir1"])

data_dir = cms_dir / "20220914-MobileNetAE/ID-targetNF"
dnn_model_dir = data_dir / "DNN" / f"{domain}_{w_and_s}" / architecture
nf_model_dir = data_dir / "NF" / f"{args.nf_type}_layers{nf_layers}" / f"{target}_{part}" / f"{domain}_{w_and_s}" / architecture / f"{files}files"
score_file_dir = data_dir / "score" / f"{target}_{part}" / f"{domain}_{w_and_s}" / architecture / f"{files}files"
summary_file_dir = data_dir / "summary" / f"{target}_{part}" / f"{domain}_{w_and_s}" / architecture
summary_file = summary_file_dir / "summary.csv"
visual_dir = data_dir / "visual" / f"{domain}_{w_and_s}" / architecture

pathlib_mkdir([dnn_model_dir, nf_model_dir, score_file_dir, summary_file_dir])

print(f"GPU DEVICES: {torch.cuda.device_count()}")

#######################################
# DNN PREPARATION
#######################################

#######################################
# list file reading
#######################################
all_list = list(all_list_dir.glob("*.list"))  # *.list
all_list = [i for i in all_list if str(i).find("._") == -1]
print(f"Total Turbines: {len(all_list)}")
for i in all_list[:]:
    if str(i).find("iwaya07") != -1 or str(i).find("mutsu-ogawara21") != -1:
        all_list.remove(i)

num_normal_turbines = len(all_list)
print(f"Normal Turbines: {num_normal_turbines}")

#######################################
# load all_list
#######################################
filename_and_ID_list = []
for i_num, i_list in enumerate(sorted(all_list)):
    filename_list = load_list_file(i_list)
    filename_and_ID = [[name, i_num] for name in filename_list]
    filename_and_ID_list.extend(filename_and_ID)

##########################################
# convert filename to spectrogram filepath
##########################################
specpath_and_ID_list = [[feature_dir / f"{i_file[0]}.pkl", i_file[1]] for i_file in filename_and_ID_list]


#######################################
# load spectrogram
#######################################
def split_spectrogram(spec):
    spec_split = np.array_split(spec, np.arange(spec.shape[1])[::width][1:], 1)
    spec_split = [np.reshape(array, (1, array.shape[0], array.shape[1])) for array in spec_split]
    if spec_split[-1].shape[2] != width:
        return spec_split[:-1]
    else:
        return spec_split


def load_spectrogram(specpath):
    if not specpath.exists():
        raise FileNotFoundError(specpath)
    spec = joblib.load(specpath)
    spec = np.array(spec, dtype=np.float32)
    if domain.find("FLAC") != -1 or domain.find("FBANK") != -1:
        spec = spec.T
    spec_list = split_spectrogram(spec)
    return spec_list


def load_spectrogram_and_ID(specpath_and_ID):
    specpath, ID = specpath_and_ID[0], specpath_and_ID[1]
    spec_list = load_spectrogram(specpath)
    spec_and_ID_list = [[spec, ID] for spec in spec_list]
    return spec_and_ID_list


with Pool(n_core) as p:
    spec_and_ID_list = unroll(list(tqdm(p.imap(load_spectrogram_and_ID, specpath_and_ID_list), total=len(specpath_and_ID_list))))

print(f"normal len: {len(spec_and_ID_list)}")

#######################################
# Preprocessing
#######################################
train_data = np.array([spec_and_ID[0] for spec_and_ID in spec_and_ID_list])
train_label = np.array([spec_and_ID[1] for spec_and_ID in spec_and_ID_list])

# (N, channel, freq, width) -> (N * width, freq) -> (N, channel, freq, width)
shape = train_data.shape
train_data = train_data.transpose((0, 1, 3, 2)).reshape(shape[0] * shape[3], shape[2])
train_scaler_model = MinMaxScaler()
train_data = train_scaler_model.fit_transform(train_data)
train_data = train_data.reshape(shape[0], shape[1], shape[3], shape[2]).transpose((0, 1, 3, 2))

train_data = torch.tensor(train_data)
train_label = torch.tensor(train_label)

train_dataset = TensorDataset(train_data, train_label)
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4)
device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")


#######################################
# Discriminator Learning
#######################################
extractor = MelMobileNetV2_IDdiscriminator10_1(bottleneck=bottleneck, mid1=mid1, id_num=num_normal_turbines, width=width)
optimizer = optim.Adam(extractor.parameters(), lr=1e-3)

if not args.noinfo:
    summary(
        model=extractor,
        input_data=[train_data[0].reshape(1, train_data.shape[1], train_data.shape[2], train_data.shape[3]), torch.tensor([[0]])],
        device=device,
    )

#######################################
# Saved Model Loading
#######################################
dnn_model_list = sorted(list(dnn_model_dir.glob("epoch*.pkl")))
if len(dnn_model_list) == 0 or args.network:
    saved_epoch = 0
else:
    saved_epoch = min(int(dnn_model_list[-1].name[5:8]), n_epoch)
    checkpoint = torch.load(dnn_model_dir / f"epoch{str(saved_epoch).zfill(3)}.pkl", map_location="cpu")
    extractor.load_state_dict(checkpoint["model"])
    extractor = extractor.to(device)
    optimizer.load_state_dict(checkpoint["optimizer"])
print(f"saved_epoch: {saved_epoch}")

extractor = extractor.to(device)


#######################################
# Training
#######################################
def train_basic_step(train_loader, extractor, criterion, optimizer, epoch):
    sum_loss = 0.0

    with tqdm(train_loader) as pbar:
        pbar.set_description(f"[Epoch {epoch}]")
        for input, label in pbar:
            input, label = input.to(device), label.to(device)
            output, bn = extractor(input, label)
            loss = criterion(output, label)
            pbar.set_postfix(OrderedDict(loss=loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss
    return sum_loss / len(train_loader)


def accuracy_calc(train_loader, extractor):
    correct_num = 0
    total_num = 0
    for input, label in train_loader:
        input, label = input.to(device), label.to(device)
        output, _ = extractor(input, label)
        predict_id = torch.argmax(output, dim=1)
        correct_num += torch.sum(predict_id == label)
        total_num += len(label)
    return correct_num.float() / total_num


criterion = nn.NLLLoss()

for i_epoch in range(saved_epoch + 1, n_epoch + 1):
    extractor.train()
    train_loss = train_basic_step(train_loader, extractor, criterion, optimizer, i_epoch)
    state = {
        "model": extractor.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state, dnn_model_dir / f"epoch{str(i_epoch).zfill(3)}.pkl")
    extractor.eval()
    with torch.no_grad():
        acc_loss = accuracy_calc(train_loader, extractor)
    print(f"{i_epoch} epoch\ttrain_loss: {train_loss:.6f}\ttrain_acc: {acc_loss:.3f}")


##################################
# Extractor evaluation mode
###################################
extractor.eval()
# device = torch.device("cpu")
extactor = extractor.to(device)

#######################################
# visualize inputS and featureS
#######################################
if args.visual:
    tsne_num = 5000
    tsne_train_list = []

    # 全クラスから少しずつサンプルをとる
    # each_wt_data_num = int(tsne_num / num_normal_turbines)
    # first_idx = 0
    # wt_ids = [data[1] for data in spec_and_ID_list]
    # for wt_num in range(num_normal_turbines):
    #     tmp_array = spec_and_ID_list[first_idx : first_idx + each_wt_data_num]
    #     tsne_train_list.extend(tmp_array)
    #     first_idx = wt_ids.index(wt_num + 1) if wt_num + 1 in wt_ids else 0

    # 最初のnクラスまででサンプルをとる
    class_num = 5
    each_wt_data_num = int(tsne_num / class_num)
    first_idx = 0
    wt_ids = [data[1] for data in spec_and_ID_list]
    for wt_num in range(class_num):
        tmp_array = spec_and_ID_list[first_idx : first_idx + each_wt_data_num]
        tsne_train_list.extend(tmp_array)
        first_idx = wt_ids.index(wt_num + 1) if wt_num + 1 in wt_ids else 0

    tsne_train = np.array([data[0] for data in tsne_train_list])
    tsne_label = np.array([data[1] for data in tsne_train_list])

    print("input t-SNE training")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=2000, verbose=1, random_state=0)
    tsne_train_flatten = tsne_train.reshape(len(tsne_train), -1)
    input_embedded = tsne.fit_transform(tsne_train_flatten)
    df = pd.DataFrame({"label": tsne_label, "feature1": input_embedded[:, 0], "feature2": input_embedded[:, 1]})
    fig = plt.figure(figsize=(15, 15))
    plt.scatter(df["feature1"], df["feature2"], c=df["label"], cmap="jet")
    plt.colorbar()
    pathlib_mkdir([visual_dir])
    fig.savefig(visual_dir / f"input_tsne_epoch{n_epoch}_{tsne_num}.png", bbox_inches="tight", pad_inches=0.1)

    print("featureS t-SNE training")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=2000, verbose=1, random_state=0)
    tsne_train = torch.tensor(tsne_train)
    tsne_loader = DataLoader(TensorDataset(tsne_train), batch_size=batchsize, shuffle=False, num_workers=4)
    bn_list = []
    for input in tsne_loader:
        input = input[0].to(device)
        with torch.no_grad():
            _, bn = extractor(input, None)
        bn_list.extend(bn.cpu().detach().numpy())
    bn_embedded = tsne.fit_transform(np.array(bn_list))
    df = pd.DataFrame({"label": tsne_label, "feature1": bn_embedded[:, 0], "feature2": bn_embedded[:, 1]})
    fig = plt.figure(figsize=(15, 15))
    plt.scatter(df["feature1"], df["feature2"], c=df["label"], cmap="jet")
    plt.colorbar()
    fig.savefig(visual_dir / f"feature_tsne_epoch{n_epoch}_{tsne_num}.png", bbox_inches="tight", pad_inches=0.1)

##########################################################
# NF PREPARATION
##########################################################

##############################
# list file reading
##############################
train_normal_filename_list = load_list_file(target_list_dir / f"cutin_train_normal.{target}.MainBearing1.list")[:files]
# train_normal_filename_list = load_list_file(target_list_dir / f"cutin_test_normal.{target}.MainBearing1.list")[:files]
test_normal_filename_list = load_list_file(target_list_dir / f"cutin_test_normal.{target}.MainBearing1.list")
test_faulty_filename_list = load_list_file(target_list_dir / f"cutin_test_faulty.{target}.MainBearing1.list")
all_filename_list = (
    load_list_file(target_list_dir / f"cutin_train_normal.{target}.MainBearing1.list")
    + load_list_file(target_list_dir / f"cutin_test_normal.{target}.MainBearing1.list")
    + load_list_file(target_list_dir / f"cutin_test_faulty.{target}.MainBearing1.list")
)

train_normal_path_list = [feature_dir / f"{i_file}.pkl" for i_file in train_normal_filename_list]
test_normal_path_list = [feature_dir / f"{i_file}.pkl" for i_file in test_normal_filename_list]
test_faulty_path_list = [feature_dir / f"{i_file}.pkl" for i_file in test_faulty_filename_list]
all_path_list = [feature_dir / f"{i_file}.pkl" for i_file in all_filename_list]


def load_header(feature_path):
    vib_path = vib_dir1 / feature_path.relative_to(feature_dir)
    header = joblib.load(vib_path)["header"]
    return (feature_path, header)


with Pool(n_core) as p:
    train_normal_path_and_header_list = list(p.imap(load_header, train_normal_path_list))
    test_normal_path_and_header_list = list(p.imap(load_header, test_normal_path_list))
    test_faulty_path_and_header_list = list(p.imap(load_header, test_faulty_path_list))
    all_path_and_header_list = list(tqdm(p.imap(load_header, all_path_list), total=len(all_path_list), desc="Loading target file"))
all_path_and_header_list = sorted(all_path_and_header_list, key=lambda path_and_header: path_and_header[1]["Time"])
# all_path_list = [path for (path, _) in all_path_and_header_list]
all_date = [header["Time"] for (_, header) in all_path_and_header_list]


#############################
# NF training
#############################
print("NF TRAINING")


##########################
# get bottleneck feature
##########################
def get_bn_features(specpath):
    spec = np.array(load_spectrogram(specpath))

    # (N, channel, freq, width) -> (N * width, freq) -> (N, channel, freq, width)
    shape = spec.shape
    spec = spec.transpose((0, 1, 3, 2)).reshape(shape[0] * shape[3], shape[2])
    # spec = pca.transform(spec)
    spec = train_scaler_model.transform(spec)
    spec = spec.reshape(shape[0], shape[1], shape[3], shape[2]).transpose((0, 1, 3, 2))

    input = torch.tensor(spec).to(device)
    with torch.no_grad():
        _, bn_features = extractor(input, None)
    return bn_features.cpu().detach().numpy()


nf_train_features = [get_bn_features(specpath) for specpath in train_normal_path_list]
nf_train_features = np.array(unroll(nf_train_features))

nf_scaler_model = MinMaxScaler()
nf_train_features = nf_scaler_model.fit_transform(nf_train_features)

nf_data_loader = DataLoader(nf_train_features, batch_size=512, shuffle=True, num_workers=4)

#############################################
# Training NF
#############################################
normal_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(bottleneck).to(device), torch.eye(bottleneck).to(device))
if args.nf_type == "ResFlow":
    nf_model = ResFlow(dims=[bottleneck], cfg={"layers": nf_layers, "estimator": "unbias"})
elif args.nf_type == "FlowPP":
    nf_model = FlowPP(dims=[bottleneck], cfg={"layers": nf_layers, "mixtures": 4})
elif args.nf_type == "RealNVP":
    nf_model = RealNVP(dims=[bottleneck], cfg={"layers": nf_layers})
else:
    raise NotImplementedError(f"This type of Flow is not implemented. {args.nf_type}")
optimizer = optim.Adam(nf_model.parameters(), lr=1e-3)
early_stopping = EarlyStopping(patience=30, verbose=1)

if not args.noinfo:
    summary(model=nf_model, input_size=(1, *nf_train_features.shape[1:]), device=device)

nf_model_list = sorted(list(nf_model_dir.glob("epoch*.pkl")))
if len(nf_model_list) == 0 or args.force:
    saved_epoch = 0
else:
    saved_epoch = min(int(nf_model_list[-1].name[5:-4]), nf_epoch)
    checkpoint = torch.load(nf_model_dir / f"epoch{str(saved_epoch).zfill(4)}.pkl", map_location="cpu")
    nf_model.load_state_dict(checkpoint["model"], strict=False)
    nf_model = nf_model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer"])
print(f"saved_epoch: {saved_epoch}")
nf_model = nf_model.to(device)


def nf_train_basic_step(train_loader, dnnmodel, optimizer):
    sum_loss = 0.0
    for input in train_loader:
        input = input.to(device)
        z, log_df_dz = dnnmodel(input)
        loss = torch.mean(-normal_dist.log_prob(z) - log_df_dz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss
    return sum_loss / len(train_loader)


pbar = tqdm(range(saved_epoch + 1, nf_epoch + 1), desc="TrainNF")
i_epoch = saved_epoch
for i_epoch in pbar:
    pbar.set_description(f"[Epoch {i_epoch}]")
    nf_model.train()
    train_loss = nf_train_basic_step(nf_data_loader, nf_model, optimizer)
    pbar.set_postfix(OrderedDict(loss=train_loss.item()))
    stopping = early_stopping(train_loss)
    if i_epoch % 10 == 0 or i_epoch == nf_epoch or stopping:
        state = {"model": nf_model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(state, nf_model_dir / f"epoch{str(i_epoch).zfill(4)}.pkl")
    if stopping:
        break
pbar.close()
nf_model.eval()


######################
# calculating score
######################
def calc_score(transformed_features):
    transformed_features = torch.from_numpy(transformed_features).to(device)
    z, log_df_dz = nf_model(transformed_features)
    log_py = normal_dist.log_prob(z) + log_df_dz
    log_py = log_py.cpu().detach().numpy()
    score = -1 * log_py
    median_score = np.median(score)
    return median_score


def scoring(path_and_header_list, desc):
    nf_train_data = unroll([[(feature, header["Time"]) for feature in get_bn_features(path)] for (path, header) in tqdm(path_and_header_list, desc="GetFeatures")])
    nf_train_data = np.array(nf_train_data, dtype=object)
    time_list = [header["Time"] for (_, header) in path_and_header_list]
    nf_train_features, nf_train_time = np.stack(nf_train_data[:, 0]), nf_train_data[:, 1]
    nf_train_features = torch.from_numpy(nf_scaler_model.transform(nf_train_features))
    nf_data_loader = DataLoader(nf_train_features, batch_size=16384, shuffle=False)

    score = np.zeros(len(nf_train_features))
    for k in range(10):
        tmp = []
        for input in tqdm(nf_data_loader, desc=desc):
            input = input.to(device)
            z, log_df_dz = nf_model(input)
            neg_log_py = -1 * (normal_dist.log_prob(z) + log_df_dz)
            neg_log_py = neg_log_py.cpu().detach().numpy()
            tmp.extend(neg_log_py)
        score += np.array(tmp)
    score /= 10

    score_by_time = []
    start_idx = 0
    for idx in range(len(nf_train_time)):
        if idx and nf_train_time[idx] != nf_train_time[idx - 1]:
            score_by_time.append(np.median(score[start_idx : idx + 1]))
            start_idx = idx
    score_by_time.append(np.median(score[start_idx:]))

    return {
        "time": time_list,
        "score": score_by_time,
    }


train_score_list = scoring(train_normal_path_and_header_list, "TrainScore")
normal_score_list = scoring(test_normal_path_and_header_list, "NormalScore")
faulty_score_list = scoring(test_faulty_path_and_header_list, "FaultyScore")
# all_score_list = scoring(all_path_and_header_list, "AllScore")

train_score = train_score_list["score"]
normal_score = normal_score_list["score"]
faulty_score = faulty_score_list["score"]
# all_score = all_score_list["score"]

train_score_df = pd.DataFrame({"Time": train_score_list["time"], "Score": train_score_list["score"]})
train_score_df["Time"] = pd.to_datetime(train_score_df["Time"])
normal_score_df = pd.DataFrame({"Time": normal_score_list["time"], "Score": normal_score_list["score"]})
normal_score_df["Time"] = pd.to_datetime(normal_score_df["Time"])
faulty_score_df = pd.DataFrame({"Time": faulty_score_list["time"], "Score": faulty_score_list["score"]})
faulty_score_df["Time"] = pd.to_datetime(faulty_score_df["Time"])


###################
# ROC
###################
scoring = ScoreSave(save_dir=score_file_dir)
fpr, tpr, threshold = scoring.eer(abnormal_score=faulty_score, normal_score=normal_score)
roc_auc = scoring.calc_auc(fpr, tpr)
fig = plt.figure(figsize=(16, 12))
plt.plot(fpr, tpr, marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
print(roc_auc)

###################
# f-measure
###################
f_value = f_value(
    train_score,
    normal_score,
    faulty_score,
    designed_percentile=args.designed_percentile,
)
accuracy4test = calc_healthy_sample_accuracy(train_score, normal_score)
faulty4test = calc_anomaly_sample_accuracy(train_score, faulty_score)

hsr = accuracy4test["Accuracy"]
asr = faulty4test["Accuracy"]


df = {
    "Files": files,
    "Bottleneck": bottleneck,
    "nf_type": args.nf_type,
    "nf_layers": nf_layers,
    "AUC": roc_auc,
    "HSR": hsr,
    "ASR": asr,
    "F": f_value["f-measure"],
    "Threshold": accuracy4test["threshold"],
    "PCA": False,
    "epoch": n_epoch,
    "nf_epoch": i_epoch,
    "mid1": mid1,
}
df = pd.DataFrame([df])

if summary_file.exists():
    df1 = pd.read_csv(summary_file)
    df = pd.concat([df1, df], ignore_index=True)
df.to_csv(summary_file, index=False)

score_dump_dir = score_file_dir / f"ROC:{roc_auc:.2f}_HSR:{hsr:.2f}_ASR:{asr:.2f}"
pathlib_mkdir([score_dump_dir])
train_score_df.to_csv(score_dump_dir / f"{target}_{part}_train_score.csv", index=False)
normal_score_df.to_csv(score_dump_dir / f"{target}_{part}_normal_score.csv", index=False)
faulty_score_df.to_csv(score_dump_dir / f"{target}_{part}_faulty_score.csv", index=False)


print(f"threshold: {accuracy4test['threshold']}")
# normal_bn_features = np.array(unroll([data["BNF"] for data in normal_score_list]))
# normal_score_label = np.array(
#     unroll([[1] * len(data["BNF"]) if data["score"] < accuracy4test["threshold"] else [2] * len(data["BNF"]) for data in normal_score_list])
# )
# faulty_bn_features = np.array(unroll([data["BNF"] for data in faulty_score_list]))
# faulty_score_label = np.array(
#     unroll([[3] * len(data["BNF"]) if data["score"] > accuracy4test["threshold"] else [4] * len(data["BNF"]) for data in faulty_score_list])
# )
# tsne_train = np.concatenate([gmm_train_features, normal_bn_features, faulty_bn_features])
# # tsne_train = np.concatenate([gmm_train_features, faulty_bn_features])
# tsne_label = np.concatenate([[0] * len(gmm_train_features), normal_score_label, faulty_score_label])
# # tsne_label = np.concatenate([[0] * len(gmm_train_features), faulty_score_label])
# if len(tsne_train) > 30000:
#     choice = np.random.choice(len(tsne_label), 30000, replace=True)
#     tsne_train = tsne_train[choice]
#     tsne_label = tsne_label[choice]

# print("target features t-SNE training")
# tsne = TSNE(n_components=2, perplexity=30, n_iter=2000, verbose=1, random_state=0)
# tsne_train = torch.tensor(tsne_train)
# bn_embedded = tsne.fit_transform(tsne_train)
# df = pd.DataFrame({"label": tsne_label, "feature1": bn_embedded[:, 0], "feature2": bn_embedded[:, 1]})
# fig = plt.figure(figsize=(15, 15))
# plt.scatter(df["feature1"], df["feature2"], c=df["label"], cmap="jet")
# plt.colorbar()
# pathlib_mkdir([visual_dir])
# fig.savefig(visual_dir / f"target_feautres_tsne_epoch{n_epoch}_{len(tsne_train)}.png", bbox_inches="tight", pad_inches=0.1)
# df.to_csv(visual_dir / "tsne_features_and_label.csv", index=False)

# print("raw target features t-SNE training")
# df = pd.DataFrame({"label": tsne_label, "feature1": tsne_train[:, 0], "feature2": tsne_train[:, 1]})
# fig = plt.figure(figsize=(15, 15))
# plt.scatter(df["feature1"], df["feature2"], c=df["label"], cmap="jet")
# plt.colorbar()
# pathlib_mkdir([visual_dir])
# fig.savefig(visual_dir / f"raw_target_feautres_tsne_epoch{n_epoch}_{len(tsne_train)}.png", bbox_inches="tight", pad_inches=0.1)

if args.visual:
    pathlib_mkdir([visual_dir])

    normal_bn_features = np.array(unroll([data["BNF"] for data in normal_score_list]))
    normal_score_label = np.array(
        unroll([[1] * len(data["BNF"]) for data in normal_score_list])
    )
    faulty_bn_features = np.array(unroll([data["BNF"] for data in faulty_score_list]))
    faulty_score_label = np.array(
        unroll([[2] * len(data["BNF"]) for data in faulty_score_list])
    )
    target_tsne_train = np.concatenate([normal_bn_features, faulty_bn_features])
    target_tsne_label = np.concatenate([normal_score_label, faulty_score_label])
    if len(target_tsne_train) > 5000:
        choice = np.random.choice(len(target_tsne_label), 5000, replace=False)
        target_tsne_train = target_tsne_train[choice]
        target_tsne_label = target_tsne_label[choice]

    print("target featureS t-SNE training")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=2000, verbose=1, random_state=0)
    tsne_train = target_tsne_train
    tsne_label = target_tsne_label
    bn_embedded = tsne.fit_transform(tsne_train)
    df = pd.DataFrame({"label": tsne_label, "feature1": bn_embedded[:, 0], "feature2": bn_embedded[:, 1]})
    df.to_csv(visual_dir / f"{architecture}_{target}_featureS.csv", index=False)
    fig = plt.figure(figsize=(8, 8))
    normal_idx = (df["label"] == 1)
    anomaly_idx = (df["label"] == 2)
    plt.scatter(df["feature1"][normal_idx], df["feature2"][normal_idx], c="green", label="normal")
    plt.scatter(df["feature1"][anomaly_idx], df["feature2"][anomaly_idx], c="red", label="anomaly")
    plt.legend()
    # plt.colorbar()
    fig.savefig(visual_dir / f"{architecture}_{target}_featureS.png", bbox_inches="tight", pad_inches=0.1)
