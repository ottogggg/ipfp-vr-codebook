if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=MSCF

root_path_name=./data/traffic/
data_path_name=traffic.csv
model_id_name=traffic_mlp
data_name=custom

seq_len=720
for pred_len in 96 192 336 720
do
  python -u ./run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 24 \
    --enc_in 862 --dec_in 862 --c_out 862 \
    --model_type mlp \
    --d_model 128 \
    --ms_num_scales 4 \
    --ms_pool_stride 2 \
    --train_epochs 10 \
    --patience 3 \
    --itr 1 --batch_size 32 --learning_rate 0.001
done
