if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=SparseTSF

root_path_name=/home/zlq/Code/SparseTSF/data/exchange_rate/
data_path_name=exchange_rate.csv
model_id_name=Exchange_rate
data_name=custom

# 720 336 336 336
seq_len=96
for pred_len in 96 192 336 720
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
    --period_len 24 \
    --enc_in 8 \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 8 --learning_rate 0.02
done

