if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=DLinear

root_path_name=/home/zlq/Code/SparseTSF/data/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

seq_len=336
for pred_len in 336
do
  python -u /home/zlq/Code/SparseTSF/run_longExp.py \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --features M \
    --label_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --freq h \
    --period_len 24 \
    --des Exp \
    --train_epochs 10 \
    --itr 1 --batch_size 128 --learning_rate 0.0001
done