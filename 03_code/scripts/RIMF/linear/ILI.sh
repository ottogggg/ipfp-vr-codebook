if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=RIMF

root_path_name=/home/zlq/Code/SparseTSF/data/ILI/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

seq_len=96
for pred_len in 24 32 48 60
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
    --period_len 16 \
    --enc_in 7 \
    --train_epochs 30 \
    --patience 5 \
    --period_list_str "[4]" \
    --itr 1 --batch_size 32 --learning_rate 0.05
done

