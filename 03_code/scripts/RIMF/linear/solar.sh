if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=RIMF

root_path_name=/home/zlq/Code/SparseTSF/data/solar/
data_path_name=solar_AL.txt
model_id_name=Solar
data_name=Solar

seq_len=336
for pred_len in 96
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
    --enc_in 137 \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 16 --learning_rate 0.02
done