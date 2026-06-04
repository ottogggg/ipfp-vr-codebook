if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=RIMF

root_path_name=/home/zlq/Code/SparseTSF/data/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
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
    --period_len 24 \
    --model_type 'mlp' \
    --d_model 128 \
    --enc_in 321 \
    --train_epochs 10 \
    --patience 5 \
    --samples_ratio 0.5 \
    --itr 3 --batch_size 32 --learning_rate 0.02
done