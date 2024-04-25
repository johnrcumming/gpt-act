# GPT-ACT

GPT-ACT is an implementation of Alex Graves' Adaptive Computation Time (ACT) applied to GPT models. It introduces a novel approach to make the transformer architecture behave like a recurrent neural network (RNN) over model layers and complexity.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

GPT-ACT is a modification of the popular GPT (Generative Pre-trained Transformer) model that incorporates the Adaptive Computation Time (ACT) mechanism proposed by Alex Graves. This modification allows the transformer to dynamically adjust the number of computational steps it takes for each input, effectively making it behave like an RNN over model layers and complexity.

## Installation

To install GPT-ACT, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/gpt-act.git`
2. Navigate to the project directory: `cd gpt-act`
3. Install the required dependencies: `pip install -r requirements.txt`


# Training
To use this script, you can run it from the command line with the appropriate arguments. For example, to train a model on the 'wikitext' dataset, you could use the following command:

```bash
python gpt2act_trainer.py --preprocess_dataset --dataset_name wikitext

python gpt2act_trainer.py --train_model --dataset_name wikitext --run_name test01

```



## Training Arguments
- `--preprocess_dataset`: Flag to trigger dataset preprocessing. Default is False.
- `--train_model`: Flag to trigger model training. Default is False.
- `--calculate_perplexity`: Flag to trigger perplexity calculation. Default is False.
- `--model_config`: Huggingface Transformers Model Config. Default is "gpt2".
- `--dataset_name`: Huggingface Datasets Name. Default is "wikitext".
- `--dataset_config`: Huggingface Datasets Configuration Name. Default is None.
- `--deepspeed_config`: Deepspeed JSON Config File. Default is None.
- `--data_dir`: Dataset Directory. Default is 'data'.
- `--log_dir`: Tensorboard Log Dir. Default is 'runs'.
- `--checkpoint_dir`: Checkpoint Save Dir. Default is 'checkpoints'.
- `--checkpoint`: Checkpoint to Continue From. Default is None.
- `--distill`: Flag to trigger model distillation. Default is False.
- `--pretrained`: Flag to copy weights from GPT2 model_config. Default is False.
- `--freeze_pretrained`: Flag to freeze pretrained weights during training. Default is False.
- `--lambda_kd`: Knowledge Distillation Loss Weight. Default is 1e-6.
- `--temperature_kd`: Knowledge Distillation temperature_kd. Default is 4.0.
- `--max_grad_norm`: Gradient Clipping Max Grad Norm. Default is 1.0.
- `--dynamic_stride`: Dynamic Block Stride. Default is None.
- `--num_procs`: Number of Processes for Dataset Processing. Default is 2.
- `--logging_steps`: Log every n steps. Default is 10.
- `--save_steps`: Save checkpoint every n steps. Default is 500.
- `--eval_steps`: Evaluate every n steps, default evaluate at epoch. Default is 0.
- `--warmup_steps`: Optimizer Warmup steps. Default is 1000.
- `--learning_rate`: Optimizer Learning Rate. Default is 1e-3.
- `--binary_embedding`: Flag to use Experimental Binary Embedding. Default is False.
- `--n_positions`: n_positions - context length. Default is 1024.
- `--train_epochs`: Training Epochs. Default is 5.
- `--train_batch_size`: Training Batch Size. Default is 16.
- `--eval_batch_size`: Evaluation Batch Size. Default is 16.
- `--gradient_accumulation_steps`: Gradient Accumulation Steps. Default is 32.
- `--parallelize`: Flag to parallelize Model - split model across GPUs. Default is False.
- `--verbose`: Flag for Verbose Logging. Default is False.
- `--stream_dataset`: Flag to Stream Dataset. Default is False.
- `--no_fp16`: Flag to disable FP16 Training. Default is True.
- `--max_steps`: Number of train steps for streaming_datasets. Default is -1.
- `--act_commitment_cost`: ACT Loss commitmemt cost. Default is 1e-3.
- `--gradient_checkpointing`: Flag to enable Gradient Checkpointing. Default is False.
- `--no_cuda`: Flag to disable CUDA. Default is False.
- `--push_to_hub_model_id`: The name of the repository to which push the Trainer. Default is None.
- `--push_to_hub_organization`: The name of the organization in with to which push the Trainer. Default is None.
- `--push_to_hub_token`: The token to use to push the model to the Hub. Default is None.
- `--report_to`: The list of integrations to report the results and logs to. Default is "all".
- `--run_name`: A descriptor for the run. Typically used for wandb logging. Default is None.
- `--halting_function_spec`: Halting Function Spec. Default is None.
- `--layerwise_attn`: Set Layerwise Attention. Default is 'simple'.
- `--no_group_texts`: Flag to disable Text Grouping. Default is True.
- `--act_depth_factor`: ACT Depth Factor. Default is None.
- `--act_depth`: ACT Depth. Default is None.