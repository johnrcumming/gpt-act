import os
import sys
import traceback
from argparse import ArgumentParser


import transformers
import datasets
#from datasets import list_datasets, load_dataset, list_metrics, load_metric, DatasetDict, Dataset

from transformers import GPT2Config
from transformers import GPT2Tokenizer
from transformers import Trainer, TrainingArguments

from gpt2_act import GPT2ACTLMHeadModel, GPT2ACTConfig

def group_texts(block_size, tokenizer=None):
    def group_texts_fn(examples):
        if tokenizer is not None:
            examples = tokenizer(examples['text'])
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys() }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    return group_texts_fn

def load_streaming_dataset(model_config, data_dir='data', dataset_name='wikitext', dataset_config=None, num_procs=10, val_size=0.05, test_size=0.05):
    """Load a streaming dataset from the datasets library, and preprocess it for training."""
    dataset_dir = os.path.join(data_dir, dataset_name)
    cache_dir = os.path.join(data_dir, 'cache')

    if dataset_config is None:
        dataset_config = datasets.get_dataset_config_names(dataset_name)[0]

    tokenizer = GPT2Tokenizer.from_pretrained(model_config)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = datasets.load_dataset(dataset_name, dataset_config, data_dir=dataset_dir, cache_dir=cache_dir, streaming=True)
    dataset = dataset.map(lambda examples: tokenizer(examples["text"]), batched=True, remove_columns=["text"])
    dataset = dataset.map(group_texts(tokenizer.model_max_length), batched=True)

    return dataset

def preprocess_dataset(model_config, data_dir='data', cache_dir=None, dataset_name='wikitext', dataset_config=None, num_procs=10, val_size=0.05, test_size=0.05):
    """Load a dataset from the datasets library, and preprocess it for training. and save dataset locally."""
    dataset_dir = os.path.join(data_dir, dataset_name)
    if cache_dir is None:
        cache_dir = os.path.join(data_dir, 'cache')

    if not os.path.exists(dataset_dir):
        if dataset_config is None:
            dataset_config = datasets.get_dataset_config_names(dataset_name)[0]

        tokenizer = GPT2Tokenizer.from_pretrained(model_config)
        tokenizer.pad_token = tokenizer.eos_token

        dataset = raw_dataset = datasets.load_dataset(dataset_name, dataset_config, data_dir=dataset_dir, cache_dir=cache_dir, num_proc=num_procs)
        dataset = tok_dataset = dataset.map(lambda examples: tokenizer(examples["text"]), num_proc=num_procs, batched=True, remove_columns=["text"])
        dataset = lm_dataset = tok_dataset.map(group_texts(tokenizer.model_max_length), num_proc=num_procs, batched=True)

        if 'validation' not in dataset:
            subdataset = dataset['train'].train_test_split(test_size=val_size, shuffle=True)
            dataset['train'] = subdataset['train']
            dataset['validation'] = subdataset['test']

        if 'test' not in dataset:
            subdataset = dataset['train'].train_test_split(test_size=test_size, shuffle=True)
            dataset['train'] = subdataset['train']
            dataset['test'] = subdataset['test']

        dataset.save_to_disk(dataset_dir)


def train(data_dir, base_logging_dir, checkpoint_dir, dataset_name,
          num_train_epochs=5, train_batch_size=2, eval_batch_size=2, gradient_accumulation_steps=256, parallelize=False,
          gradient_checkpointing=False, act_commitment_cost=1e-3,
          model_config='gpt2-xl', pretrained_weights=None, checkpoint=None, verbose=True, fp16=False, 
          stream_dataset=False, max_steps=-1, storage_options=None, num_procs=10):
    
    """Train a GPT2ACT model on a dataset."""

    if verbose:
        transformers.utils.logging.set_verbosity_info()


    if stream_dataset:
        dataset = load_streaming_dataset(model_config, data_dir=data_dir, dataset_name=dataset_name)
        train_dataset = dataset['train']
        val_dataset = dataset['validation'] if 'validation' in dataset else None

    else:
        dataset_dir=os.path.join(data_dir, dataset_name)
        dataset = datasets.DatasetDict.load_from_disk(dataset_dir)
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        
        

    gpt2_config = GPT2Config.from_pretrained(model_config)
    config = GPT2ACTConfig(act_commitment_cost=act_commitment_cost, gradient_checkpointing=gradient_checkpointing, **gpt2_config.to_dict())
    model = GPT2ACTLMHeadModel(config)

    os.makedirs(base_logging_dir, exist_ok=True)
    run_dir = dataset_name + str(len(os.listdir(base_logging_dir)))
    logging_dir = os.path.join(base_logging_dir, run_dir)

    training_args = TrainingArguments(
        output_dir= checkpoint_dir,          # output directory
        logging_dir= logging_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,              # total # of training epochs
        per_device_train_batch_size=train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=eval_batch_size if eval_batch_size > 0 else None,   # batch size for evaluation
        warmup_steps=5000,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        save_total_limit=5,
        logging_steps=10,
        evaluation_strategy='epoch' if eval_batch_size > 0 else 'no',
        save_strategy='steps',
        save_steps=500,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_accumulation_steps=gradient_accumulation_steps,
        ignore_data_skip=True,
        fp16=fp16,
        max_steps=max_steps,
        dataloader_pin_memory=True,
        do_train=True,
        do_eval=eval_batch_size > 0,
        dataloader_num_workers=num_procs
    )

    if parallelize:
        model.parallelize()
    
    if pretrained_weights is not None:
        model.copyweights(pretrained_weights)

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # evaluation dataset
    )

    try:
        trainer.train(checkpoint)
        trainer.save_model(os.path.join(training_args.output_dir, 'best'))
    except KeyboardInterrupt:
        trainer.save_model(os.path.join(training_args.output_dir, 'interrupt'))
    except Exception as e:
        traceback.print_exc()
        trainer.save_model(os.path.join(training_args.output_dir, 'crash'))


def main():
    parser = ArgumentParser(description='Perform GPT2ACT Dataset Build and Training.', add_help=False)

    parser.add_argument('--preprocess_dataset', default=False, action='store_true', help='Build Dataset.')
    parser.add_argument('--train_model', default=False, action='store_true', help='Train Model.')

    parser.add_argument('--model_config', type=str, default="gpt2-xl", help='Huggingface Transformers Model Config.')
    parser.add_argument('--dataset_name', type=str, default="openwebtext", help='Huggingface Datasets Name.')
    parser.add_argument('--dataset_config', type=str, default=None, help='Huggingface Datasets Configuration Name.')

    parser.add_argument('--data_dir', type=str, default='data', help='Dataset Directory.')
    parser.add_argument('--log_dir', type=str, default='runs', help='Tensorboard Log Dir.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Ceckpoint Save Dir.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Ceckpoint to Continue From.')

    parser.add_argument('--num_procs', type=int, default=10, help='Number of Processes for Dataset Processing.')
    parser.add_argument('--train_epochs', type=int, default=5, help='Training Epochs.')
    parser.add_argument('--train_batch_size', type=int, default=2, help='Training Batch Size.')
    parser.add_argument('--eval_batch_size', type=int, default=2, help='Evaluation Batch Size.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=256, help='Gradient Accumulation Steps.')
    parser.add_argument('--parallelize', default=False, action='store_true', help='Parallelize Model - split model across GPUs.')
    parser.add_argument('--verbose', default=False, action='store_true', help='Verbose Logging.')
    parser.add_argument('--stream_dataset', default=False, action='store_true', help='Stream Dataset.')
    parser.add_argument('--fp16', default=False, action='store_true', help='FP16 Training.')
    parser.add_argument('--max_steps', type=int, default=-1, help='Number of train steps for streaming_datasets.')
    parser.add_argument('--act_commitment_cost', type=float, default=1e-3, help='ACT Loss commitmemt cost.')
    parser.add_argument('--gradient_checkpointing', default=False, action='store_true', help='Enable Gradient Chackpointing.')


    args = parser.parse_args()

    if args.preprocess_dataset:
        preprocess_dataset(args.model_config, data_dir=args.data_dir, 
                           dataset_name=args.dataset_name, dataset_config=args.dataset_config, num_procs=args.num_procs)

    if args.train_model:
        train(args.data_dir, args.log_dir, args.checkpoint_dir, args.dataset_name,
                num_train_epochs=args.train_epochs, train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps, parallelize=args.parallelize,
                gradient_checkpointing=args.gradient_checkpointing, act_commitment_cost=args.act_commitment_cost,
                model_config=args.model_config, pretrained_weights=None, checkpoint=args.checkpoint, 
                verbose=args.verbose, stream_dataset=args.stream_dataset, fp16=args.fp16, max_steps=args.max_steps, num_procs=args.num_procs)

if __name__ == "__main__":
    main()
