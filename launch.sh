#!/bin/sh

#tensorboard --bind_all --port=6006 --logdir gpt-act/runs &
#jupyter lab --ip=0.0.0.0 --allow-root --port=8888 

python gpt2act_trainer.py --preprocess_dataset --dataset_name openwebtext
accelerate launch gpt2act_trainer.py --train_model --dataset_name openwebtext --data_dir d:\datasets --train_batch_size 16 --gradient_accumulation_steps 64 --learning_rate 1e-5 --run_name owt_bs1000_run1
