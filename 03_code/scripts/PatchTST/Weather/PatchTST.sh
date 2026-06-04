if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=PatchTST

root_path_name=/home/zlq/Code/SparseTSF/data/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

seq_len=336
for pred_len in 720
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
    --train_epochs 8 \
    --itr 1 --batch_size 128 --learning_rate 0.0001
done

#
#python3 /home/lch/ATFNet/run.py \
#--data_path /home/lch/ATFNet/data/ETT/ETTh1.csv \
#--model_id ETTh1_96_192 \
#--model PatchTST \
#--data ETTh1 \
#--features M \
#--seq_len 96 \
#--label_len 24 \
#--pred_len 192 \
#--e_layers 2 \
#--d_layers 1 \
#--enc_in 7 \
#--dec_in 7 \
#--c_out 7 \
#--freq h \
#--des Exp \
#--itr 1 \
#--train_epochs 10 \
#--batch_size 128 \
#
#python3 /home/lch/ATFNet/run.py \
#--data_path /home/lch/ATFNet/data/ETT/ETTh1.csv \
#--model_id ETTh1_96_336 \
#--model PatchTST \
#--data ETTh1 \
#--features M \
#--seq_len 96 \
#--label_len 24 \
#--pred_len 336 \
#--e_layers 2 \
#--d_layers 1 \
#--enc_in 7 \
#--dec_in 7 \
#--c_out 7 \
#--freq h \
#--des Exp \
#--itr 1 \
#--train_epochs 10 \
#--batch_size 128 \
#
#python3 /home/lch/ATFNet/run.py \
#--data_path /home/lch/ATFNet/data/ETT/ETTh1.csv \
#--model_id ETTh1_96_720 \
#--model PatchTST \
#--data ETTh1 \
#--features M \
#--seq_len 96 \
#--label_len 24 \
#--pred_len 720 \
#--e_layers 2 \
#--d_layers 1 \
#--enc_in 7 \
#--dec_in 7 \
#--c_out 7 \
#--freq h \
#--des Exp \
#--itr 1 \
#--train_epochs 10 \
#--batch_size 128 \


