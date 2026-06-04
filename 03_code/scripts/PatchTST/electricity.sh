if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=PatchTST

root_path_name=/home/zlq/Code/SparseTSF/data/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

seq_len=96
for pred_len in 96 192
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
    --enc_in 321 \
    --train_epochs 100 \
    --patience 5 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --lradj 'TST'\
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --itr 1 --batch_size 8 --learning_rate 0.001
done
