python inference.py \
  --scada_csv /mnt/iot-qnap5/miyauchi/data/paired_data/23-04-02-8.csv \
  --tabnet_ckpt /mnt/iot-qnap5/miyauchi/model/haenkaze/tabnet-pretrain-out2023-40dim/pretrained.pth \
  --cap_ckpt_dir /mnt/iot-qnap5/miyauchi/prefix_tuning/checkpoints/paired_data-gpt_medium-transformer-prefix-ep20-bs8-lr0.001 \
  --beam_size 10
