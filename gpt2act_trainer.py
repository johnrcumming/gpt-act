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

from gpt2_act import GPT2ACTLMHeadModel


def preprocess_dataset(model_config, data_dir='data', cache_dir='data/cache' , dataset_name='wikitext', dataset_config=None, num_procs=10, val_size=0.05, test_size=0.05, debug=False):
    dataset_dir = os.path.join(data_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        if dataset_config is None:
            dataset_config = datasets.get_dataset_config_names(dataset_name)[0]

        raw_dataset = datasets.load_dataset(dataset_name, dataset_config, data_dir=dataset_dir, cache_dir=cache_dir)
        print(raw_dataset)
        tokenizer = GPT2Tokenizer.from_pretrained(model_config)
        tokenizer.pad_token = tokenizer.eos_token

        block_size = tokenizer.model_max_length
        def tokenize_function(examples):
            return tokenizer(examples["text"])

        dataset = tok_dataset = raw_dataset.map(lambda examples: tokenizer(examples["text"]), num_proc=num_procs, batched=True, remove_columns=["text"])

        def group_texts(examples):
        #    examples = tokenizer(examples['text'])
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

        dataset = lm_dataset = tok_dataset.map(group_texts, num_proc=num_procs, batched=True)

        if 'validation' not in dataset:
            subdataset = dataset['train'].train_test_split(test_size=val_size, shuffle=True)
            dataset['train'] = subdataset['train']
            dataset['validation'] = subdataset['test']

        if 'test' not in dataset:
            subdataset = dataset['train'].train_test_split(test_size=val_size, shuffle=True)
            dataset['train'] = subdataset['train']
            dataset['test'] = subdataset['test']

        dataset.save_to_disk(dataset_dir)

        try:
            raw_dataset.cleanup_cache_files()
            tok_dataset.cleanup_cache_files()
            if dataset != lm_dataset:
                lm_dataset.cleanup_cache_files()
        except:
            pass

def train(data_dir, base_logging_dir, checkpoint_dir, dataset_name,
          num_train_epochs=5, train_batch_size=2, eval_batch_size=2, gradient_accumulation_steps=256, parallelize=False,
          model_config='gpt2-xl', pretrained_weights=None, checkpoint=None, verbose=True):

    dataset_dir = os.path.join(data_dir, dataset_name)
    
    if verbose:
        transformers.utils.logging.set_verbosity_info()

    dataset = datasets.DatasetDict.load_from_disk(dataset_dir)
    train_dataset = dataset['train']
    val_dataset =  dataset['validation']

    config = GPT2Config.from_pretrained(model_config)
    model = GPT2ACTLMHeadModel(model_config)

    os.makedirs(base_logging_dir, exist_ok=True)
    run_dir = dataset_name + str(len(os.listdir(base_logging_dir)))
    logging_dir = os.path.join(base_logging_dir, run_dir)

    training_args = TrainingArguments(
        output_dir= checkpoint_dir,          # output directory
        logging_dir= logging_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,              # total # of training epochs
        per_device_train_batch_size=train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=eval_batch_size,   # batch size for evaluation
        warmup_steps=5000,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        save_total_limit=5,
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=500,
        load_best_model_at_end=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ignore_data_skip=True,
        fp16=True,
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

    parser.add_argument('--data_dir', type=str, default='/content/data', help='Dataset Directory.')
    parser.add_argument('--cache_dir', type=str, default='/content/data/cache', help='Dataset Cache Dir.')
    parser.add_argument('--log_dir', type=str, default='/content/data/runs', help='Tensorboard Log Dir.')
    parser.add_argument('--checkpoint_dir', type=str, default='/content/data/checkpoints', help='Ceckpoint Save Dir.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Ceckpoint to Continue From.')

    parser.add_argument('--num_procs', type=int, default=10, help='Number of Processes for Dataset Processing.')
    parser.add_argument('--train_epochs', type=int, default=5, help='Training Epochs.')
    parser.add_argument('--train_batch_size', type=int, default=2, help='Training Batch Size.')
    parser.add_argument('--eval_batch_size', type=int, default=2, help='Evaluation Batch Size.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=256, help='Gradient Accumulation Steps.')
    parser.add_argument('--parallelize', default=False, action='store_true', help='Parallelize Model - split model across GPUs.')
    parser.add_argument('--verbose', default=False, action='store_true', help='Verbose Logging.')


    args = parser.parse_args()

    if args.preprocess_dataset:
        preprocess_dataset(args.model_config, data_dir=args.data_dir, cache_dir=args.cache_dir, 
                           dataset_name=args.dataset_name, dataset_config=args.dataset_config, num_procs=args.num_procs)

    if args.train_model:
        train(args.data_dir, args.log_dir, args.checkpoint_dir, args.dataset_name,
                num_train_epochs=args.train_epochs, train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps, parallelize=args.parallelize,
                model_config=args.config, pretrained_weights=None, checkpoint=args.checkpoint, verbose=args.verbose)

if __name__ == "__main__":
    main()
