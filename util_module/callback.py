import logging
from pytorch_tabnet.callbacks import Callback

class LogCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")  # loss
        val_loss = logs.get("val_0_unsup_loss_numpy")
        logging.info(f"epoch {epoch+1 :>3} | loss:{loss:10.4f},  val_0_unsup_loss_numpy:{val_loss:10.4f}")

def set_log(filepath):
    logging.basicConfig(
        filename=filepath,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    print(f"log: {filepath}")