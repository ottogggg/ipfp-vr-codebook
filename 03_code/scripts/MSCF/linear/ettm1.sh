if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=MSCF

root_path_name=./data/ETT/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

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
    --enc_in 7 --dec_in 7 --c_out 7 \
    --model_type linear \
    --ms_num_scales 4 \
    --ms_pool_stride 2 \
    --train_epochs 20 \
    --patience 5 \
    --itr 1 --batch_size 128 --learning_rate 0.002
done
