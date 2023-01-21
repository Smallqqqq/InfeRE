#!/bin/sh
for id in AST PostScratachpad
do
python run_train.py \
    --model_name_or_path t5-base \
    --do_train \
    --do_eval \
    --seed 1 \
    --source_lang en \
    --target_lang en \
    --source_prefix "translate English to English: " \
    --train_file data/turk/"$id"_train.json \
    --validation_file data/turk/"$id"_val.json \
    --output_dir t5-base_models/seed1_"$id"_bs12_withlog \
    --per_device_train_batch_size=12 \
    --per_device_eval_batch_size=32 \
    --overwrite_output_dir \
    --predict_with_generate \
    --eval_steps=1 \
    --evaluation_strategy="epoch" \
    --num_train_epochs=30 \
    --logging_strategy="steps" \
    --logging_steps="10" \
    --save_strategy="epoch" \
    --load_best_model_at_end 
done