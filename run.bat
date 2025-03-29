rem Runner for gpt2act training
rem
accelerate launch gpt2act_trainer.py --dataset_name openwebtext --data_dir d:\datasets --train_batch_size 512 --gradient_accumulate_steps 1 --learning_rate 1e-5 --run_name owt_bs1000_run1