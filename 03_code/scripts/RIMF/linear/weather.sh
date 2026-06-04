if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=RIMF

root_path_name=/home/zlq/Code/SparseTSF/data/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

seq_len=720
for pred_len in 720
do
  python -u /home/zlq/Code/SparseTSF/run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 4 \
    --enc_in 21 \
    --train_epochs 10 \
    --model_type 'mlp' \
    --d_model 128 \
    --patience 5 \
    --samples_ratio 0.3624 \
    --period_list_str "[4,12,24]" \
    --itr 1 --batch_size 256 --learning_rate 0.02
done
