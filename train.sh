#!/bin/sh

python gpt2act_trainer.py --preprocess_dataset \
    --dataset wikitext

accelerate launch gpt2act_trainer.py --train_model \
    --dataset wikitext \
    --train_batch_size=6 --gradient_accumulation_steps=512 \
    --run_name gpt2actmoe \
    --logging_steps=5

