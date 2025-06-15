export CUDA_VISIBLE_DEVICES=0,1

python train.py \
  --dataset_name paired_data \
  --datasets_dpath /mnt/iot-qnap5/miyauchi/data \
  --rinna_gpt_name gpt_medium \
  --per_gpu_train_batch_size 24 \
  --per_gpu_eval_batch_size 24 \
  --lr 2e-5 \
  --save_every 1 \
  --mapping_type transformer \
  --prefix_length 10 \
  --prefix_dim 240 \
  --num_layers 4 \
  --n_gpu 2 \
  --only_prefix
