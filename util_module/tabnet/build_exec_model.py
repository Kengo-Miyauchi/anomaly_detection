from pytorch_tabnet.pretraining import TabNetPretrainer
import torch
import yaml
from util_module.callback import LogCallback
import logging
import os
from matplotlib import pyplot as plt

class ExecModel:
    def __init__(self, device, config, dataset_name, model_name, X_train=None, X_valid=None, refit=False,
                 feature_dim=None, n_steps=None, optimizer_params=None, batch_size_pre=None, batch_size_tr=None,
                 covariance_type=None, pretraining_ratio=None, max_epochs=None,path_to_pretrained=None,
                 mask_by_table=False, use_self_attn=False, sequence_length=None):
        self.device = device
        self.config = config
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.X_train = X_train
        self.X_valid = X_valid
        self.refit = refit
        self.feature_dim = feature_dim
        self.n_steps = n_steps
        self.optimizer_params = optimizer_params
        self.batch_size_pre = batch_size_pre
        self.batch_size_tr = batch_size_tr
        self.covariance_type = covariance_type
        self.pretraining_ratio = pretraining_ratio
        self.max_epochs = max_epochs
        self.path_to_pretrained = path_to_pretrained
        self.mask_by_table = mask_by_table
        self.use_self_attn = use_self_attn
        self.sequence_length = sequence_length
        
        # configとlogの設定
        self.set_params_from_file(config)
        self.conditions=self.gather_conditions()
        self.out_dir = self.set_out_dir()
        os.makedirs(self.out_dir, exist_ok=True)
        if(self.path_to_pretrained==None):
            self.path_to_pretrained = f'/mnt/iot-qnap5/miyauchi/model/{dataset_name}/{model_name}-{str(self.feature_dim)}dim'
            if self.mask_by_table:
                self.path_to_pretrained += f"-{self.sequence_length}seq"
        os.makedirs(self.path_to_pretrained, exist_ok=True)
        
        if(os.path.exists(self.path_to_pretrained+"/pretrained.pth")):
            print(f"Load model from {self.path_to_pretrained}/pretrained.pth")
            self.unsupervised_model=torch.load(self.path_to_pretrained+"/pretrained.pth",weights_only=False)
            if(self.refit):
                self.set_log(filepath=self.path_to_pretrained + '/pretraining_refit.log')
                #self.unsupervised_model=self.set_unsupervised_model()
                logging.info("Starting TabNet pretraining...")
                self.fit_model()
                logging.info("TabNet pretraining finished.")
                torch.save(self.unsupervised_model, self.path_to_pretrained+"/pretrained_refit.pth")
        else:
            self.set_log(filepath=self.path_to_pretrained + '/pretraining.log')
            self.unsupervised_model=self.set_unsupervised_model()
            logging.info("Starting TabNet pretraining...")
            self.fit_model()
            logging.info("TabNet pretraining finished.")
            torch.save(self.unsupervised_model, self.path_to_pretrained+"/pretrained.pth")
            # self.unsupervised_model.save_model(path_to_pretrained)
    
    # 事前学習モデルの設定
    def set_unsupervised_model(self):
        unsupervised_model = TabNetPretrainer(
            device_name=self.device,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=self.optimizer_params,
            mask_type='entmax', # "sparsemax",
            n_d = self.feature_dim,
            n_a = self.feature_dim,
            n_steps=self.n_steps,
            verbose=100,
            #warm_start=self.refit,
        )
        return unsupervised_model
    
    # 事前学習
    def fit_model(self):
        self.unsupervised_model.fit(
            X_train=self.X_train,
            eval_set=[self.X_valid],
            max_epochs=self.max_epochs , patience=1000000,
            batch_size=self.batch_size_pre, virtual_batch_size=128,
            pretraining_ratio=self.pretraining_ratio,
            callbacks=[LogCallback()],
            mask_by_table=self.mask_by_table,
            use_self_attn = self.use_self_attn,
            sequence_length = self.sequence_length,
        )
    
    # ログの設定
    def set_log(self,filepath):
        logging.basicConfig(
            filename=filepath,
            filemode='w',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        print(f"log file : {filepath}")
    
    def set_out_dir(self):
        out_dir = "result/" + self.dataset_name + "/" + self.model_name + "/" + str(self.feature_dim) + "dim"
        if(self.covariance_type!=None):
            if(self.covariance_type!='full'):
                out_dir = out_dir + "-" + self.covariance_type
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
        
    def set_params_from_dict(self, params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self 

    def set_params_from_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                params = yaml.safe_load(f)
                self.set_params_from_dict(params)
        except FileNotFoundError:
            print(f"Error: Config file not found at {file_path}")
        except yaml.YAMLError as e:
            print(f"Error: Failed to parse YAML file {file_path}: {e}")
        return self

    def gather_conditions(self):
        conditions = [("feature_dim",self.feature_dim),("optimizer_params",self.optimizer_params),("pretraining_ratio",self.pretraining_ratio),("max_epochs",self.max_epochs)]
        return conditions
    
    def train_curve(self, start=1):
        start = 1

        # 学習履歴から再構成ロスと検証セットのロスを取得
        train_loss = self.unsupervised_model.history['loss']
        valid_loss = self.unsupervised_model.history['val_0_unsup_loss_numpy']
        train_loss = train_loss[start-1:]
        valid_loss = valid_loss[start-1:]

        # グラフの作成
        plt.figure(figsize=(10, 6))
        plt.plot(range(start, start+len(train_loss)), train_loss, label="Train Loss")
        plt.plot(range(start, start+len(valid_loss)), valid_loss, label="Validation Loss")
        plt.title("Reconstruction Loss during Pretraining", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend()
        img_path = self.path_to_pretrained+"/train_curve_"+str(start)+"-"+str(start+len(train_loss)-1)+".png"
        plt.savefig(img_path)
        print(f"Train curve: {img_path}")