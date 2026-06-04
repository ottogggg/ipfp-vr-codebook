if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=RIMF

root_path_name=/home/zlq/Code/SparseTSF/data/ETT/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

# seq 720 pre 336 period 48

seq_len=1440
for pred_len in 336
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
    --period_len 48 \
    --enc_in 7 \
    --train_epochs 10 \
    --patience 5 \
    --samples_ratio 0.4 \
    --itr 1 --batch_size 256 --learning_rate 0.02
done


#每次eporch计算平均相似度