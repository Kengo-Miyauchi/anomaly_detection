python inference.py \
  --scada_csv /../data/paired_data/23-04-02-8.csv \
  --tabnet_ckpt /../model/haenkaze/tabnet-pretrain-out2023-40dim/pretrained.pth \
  --cap_ckpt_dir /../model/haenkaze/prefix_tuning/checkpoints/paired_data-gpt_medium-transformer-prefix-ep20-bs128-lr0.001 \
  --beam_size 10
