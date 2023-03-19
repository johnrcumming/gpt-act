import os
import sys
import traceback
import torch

pretrained='gpt2'
num_procs=10
block_size =  128

dataset_short_name = 'wt103'
dataset_name = ['wikitext', 'wikitext-103-v1']
dataset_splits =['train', 'test', 'validation']
dataset_dir = '/datasets/Huggingface/wikitext103_new'
checkpoint = 'checkpoints/gpt2/wt103-5/checkpoint-56500'

import transformers
transformers.utils.logging.set_verbosity_info()

from datasets import load_dataset, DatasetDict

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(pretrained)
tokenizer.pad_token = tokenizer.eos_token

if not os.path.exists(dataset_dir):
    raw_datasets = [load_dataset(*dataset_name, split=split) for split in dataset_splits ]

    if not block_size:
        block_size = tokenizer.model_max_length

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tok_datasets = [dataset.map(tokenize_function, num_proc=num_procs, batched=True, remove_columns=["text"]) for dataset in raw_datasets]

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

    datasets = [dataset.map(group_texts, num_proc=num_procs, batched=True) for dataset in tok_datasets]

    dataset = DatasetDict({split: dataset for split, dataset in zip(dataset_splits, datasets)})
    dataset.save_to_disk(dataset_dir)
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

else:
    dataset = DatasetDict.load_from_disk(dataset_dir)
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

print(train_dataset, val_dataset, test_dataset)


from transformers import AutoTokenizer, AutoConfig

from transformers import GPT2Config
config = GPT2Config.from_pretrained(pretrained)


from models.gpt2_act import GPT2ACTLMHeadModel

model = GPT2ACTLMHeadModel.from_pretrained(checkpoint)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    per_device_eval_batch_size=8,   # batch size for evaluation
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    eval_dataset=test_dataset,            # evaluation dataset
)

results = trainer.evaluate()

print(results)

import random
random.seed()
ix = random.randint(0,len(test_dataset))
inputs = test_dataset[ix]

input_ids = inputs['input_ids'][0:10]

print('Input:', tokenizer.decode(input_ids, skip_special_tokens=True))

# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
sample_outputs = model.generate(
    torch.tensor(input_ids, device=model.device).unsqueeze(0),
    do_sample=True, 
    max_length=128, 
    top_k=50, 
    top_p=0.95, 
    num_return_sequences=3
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
