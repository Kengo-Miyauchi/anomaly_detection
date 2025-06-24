export CUDA_VISIBLE_DEVICES=0,1

python train.py \
  --dataset_name paired_data \
  --datasets_dpath /mnt/iot-qnap5/miyauchi/data \
  --rinna_gpt_name gpt_medium \
  --per_gpu_train_batch_size 128 \
  --per_gpu_eval_batch_size 128 \
  --lr 1e-3 \
  --save_every 1 \
  --mapping_type transformer \
  --prefix_length 10 \
  --prefix_dim 240 \
  --num_layers 4 \
  --n_gpu 2 \
  --epochs 20 \
  --only_prefix
