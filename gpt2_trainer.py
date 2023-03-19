import os
import sys
import traceback

pretrained='gpt2'
num_procs=10

dataset_short_name = 'wt103'
dataset_name = ['wikitext', 'wikitext-103-v1']
dataset_split='train'
dataset_dir = '/datasets/Huggingface/wikitext103'

import transformers
transformers.utils.logging.set_verbosity_info()

from datasets import load_dataset, DatasetDict


if not os.path.exists(dataset_dir):

    train_raw_dataset = load_dataset(*dataset_name, split='train')
    val_raw_dataset = load_dataset(*dataset_name, split='validation')

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained)
    tokenizer.pad_token = tokenizer.eos_token

    # block_size = tokenizer.model_max_length
    block_size = 128
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    train_tok_dataset = train_raw_dataset.map(tokenize_function, num_proc=4, batched=True, remove_columns=["text"])
    val_tok_dataset = val_raw_dataset.map(tokenize_function, num_proc=4, batched=True, remove_columns=["text"])

    def group_texts(examples):
    #    examples = tokenizer(examples['text'])
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
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

    train_dataset = train_tok_dataset.map(group_texts, num_proc=num_procs, batched=True)
    val_dataset = val_tok_dataset.map(group_texts, num_proc=num_procs, batched=True)

    dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset})
    dataset.save_to_disk(dataset_dir)

else:
    dataset = DatasetDict.load_from_disk(dataset_dir)
    train_dataset = dataset['train']
    test_dataset = dataset['validation']

print(train_dataset, test_dataset)

from transformers import GPT2Config
config = GPT2Config.from_pretrained(pretrained)

from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel(config)

from transformers import Trainer, TrainingArguments

base_logging_dir = os.path.join('./runs', pretrained + 'HF')
os.makedirs(base_logging_dir, exist_ok=True)
run_dir = dataset_short_name + '-' + str(len(os.listdir(base_logging_dir)))
logging_dir = os.path.join(base_logging_dir, run_dir)
checkpoint_dir = os.path.join('./checkpoints', pretrained + 'HF', run_dir)

training_args = TrainingArguments(
    output_dir= checkpoint_dir,          # output directory
    logging_dir= logging_dir,
    overwrite_output_dir=True,
    num_train_epochs=5,              # total # of training epochs
    per_device_train_batch_size=10,  # batch size per device during training
    per_device_eval_batch_size=10,   # batch size for evaluation
    warmup_steps=5000,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    save_total_limit=5,
    logging_steps=50,
    evaluation_strategy='steps',
    eval_steps=500,
    load_best_model_at_end=True,
    gradient_accumulation_steps=6,
    ignore_data_skip=True
#    fp16=True,
)

#model.parallelize()
#model.copyweights(pretrained)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,            # evaluation dataset
)

try:
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, 'best'))
except KeyboardInterrupt:
    trainer.save_model(os.path.join(training_args.output_dir, 'interrupt'))
except Exception as e:
    traceback.print_exc()
    trainer.save_model(os.path.join(training_args.output_dir, 'crash'))
