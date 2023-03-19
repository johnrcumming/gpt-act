import os
import sys
import torch


#pretrained='gpt2-xl'
# dataset_name = ['openwebtext']
# dataset_split='train'
# dataset_dir = '/datasets/Huggingface/openwebtext'
# num_procs=10

pretrained='gpt2-large'
dataset_name = ['wikitext', 'wikitext-2-v1']
dataset_split='train'
dataset_dir = '/data/Datasets/Huggingface/wikitext2_teacher-gpt2-large'
num_procs=1

from datasets import list_datasets, load_dataset, list_metrics, load_metric, DatasetDict

if not os.path.exists(dataset_dir):
    train_raw_dataset = load_dataset(*dataset_name, split=dataset_split)

    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained)
    tokenizer.pad_token = tokenizer.eos_token

    teacher = GPT2LMHeadModel.from_pretrained(pretrained).to('cuda:0')
    teacher.eval()

    # block_size = tokenizer.model_max_length
    block_size = 128
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    train_tok_dataset = train_raw_dataset.map(tokenize_function, num_proc=num_procs, batched=True, remove_columns=["text"])

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
        if len(result["input_ids"]) > 0:
            input_ids=torch.tensor(result['input_ids'], device=teacher.device)
            labels=torch.tensor(result['labels'], device=teacher.device)

            with torch.no_grad():
                outputs = teacher(input_ids=input_ids, labels=labels)

            result['teacher_logits'] = outputs.logits.cpu().numpy()
        else:
            result['teacher_logits'] = []
            
        return result

    train_lm_dataset = train_tok_dataset.map(group_texts, num_proc=num_procs, batched=True, batch_size=20)

    dataset = train_lm_dataset.train_test_split(test_size=0.05, shuffle=True)

    dataset.save_to_disk(dataset_dir)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    train_tok_dataset.cleanup_cache_files()
    train_lm_dataset.cleanup_cache_files()

dataset = DatasetDict.load_from_disk(dataset_dir)
train_dataset = dataset['train']
test_dataset = dataset['test']

from transformers import AutoTokenizer, AutoConfig

from transformers import GPT2Config, GPT2LMHeadModel
config = GPT2Config.from_pretrained(pretrained)

from models.gpt2_act import GPT2ACTLMHeadModel, GPT2ACTDistilation

student = GPT2ACTLMHeadModel(config)

model = GPT2ACTDistilation(student, teacher, lambda_kd=1e-4, temperature=4.0)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./checkpoints/gpt2actdistil-large',          # output directory
    logging_dir='./runs/gpt2actdistil-large/run-3',
    overwrite_output_dir=True,
    num_train_epochs=10,              # total # of training epochs
    per_device_train_batch_size=3,  # batch size per device during training
    per_device_eval_batch_size=3,   # batch size for evaluation
    warmup_steps=2000,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    save_total_limit=5,
    logging_steps=5,
    evaluation_strategy='steps',
    eval_steps=500,
    load_best_model_at_end=True,
    gradient_accumulation_steps=15,
#    fp16=True,
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,            # evaluation dataset
)

try:
    trainer.train()
except:
    pass

model.student.save_pretrained(os.path.join(training_args.output_dir, 'best'))