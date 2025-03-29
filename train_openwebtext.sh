#!/bin/sh

python gpt2act_trainer.py --preprocess_dataset \
    --dataset openwebtext

accelerate config --multi_gpu \
    --num_processes 4 \
    --num_machines 1 \
    --mixed_precision fp16 \
    --distributed_type MULTI_GPU

accelerate launch gpt2act_trainer.py --train_model \
    --dataset openwebtext \
    --train_batch_size=24 \
    --eval_batch_size=24 \
    --gradient_accumulation_steps=512 \
    --run_name gpt2actmoe \
    --logging_steps=5

