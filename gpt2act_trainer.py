import os
import sys
import traceback

pretrained='gpt2-xl'
num_procs=10

dataset_short_name = 'wt103'
dataset_name = ['wikitext', 'wikitext-103-v1']
dataset_splits=['train', 'validation', 'test']
dataset_dir = '/content/gptact/data/wikitext103'
cache_dir = '/content/gptact/data/cache'
#checkpoint = 'checkpoints/gpt2/wt103-6/checkpoint-7000'
checkpoint = None

import transformers
transformers.utils.logging.set_verbosity_info()

from datasets import load_dataset, DatasetDict


from transformers import GPT2Tokenizer

def main():
  tokenizer = GPT2Tokenizer.from_pretrained(pretrained)
  tokenizer.pad_token = tokenizer.eos_token

  if not os.path.exists(dataset_dir):
      raw_datasets = [load_dataset(*dataset_name, data_dir=dataset_dir, cache_dir=cache_dir, split=split) for split in dataset_splits ]

#      if not block_size:
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

  from transformers import GPT2Config
  config = GPT2Config.from_pretrained(pretrained)

  from gpt2_act import GPT2ACTLMHeadModel

  model = GPT2ACTLMHeadModel(config)

  from transformers import Trainer, TrainingArguments

  base_logging_dir = os.path.join('./runs', pretrained)

  if checkpoint:
      checkpoint_dir = os.path.split(checkpoint)[0]
      run_dir = os.path.split(checkpoint_dir)[1]
      logging_dir = os.path.join(base_logging_dir, run_dir)

  else:
      os.makedirs(base_logging_dir, exist_ok=True)
      run_dir = dataset_short_name + '-' + str(len(os.listdir(base_logging_dir)))
      logging_dir = os.path.join(base_logging_dir, run_dir)
      checkpoint_dir = os.path.join('./checkpoints', pretrained, run_dir)

  training_args = TrainingArguments(
      output_dir= checkpoint_dir,          # output directory
      logging_dir= logging_dir,
      overwrite_output_dir=True,
      num_train_epochs=5,              # total # of training epochs
      per_device_train_batch_size=2,  # batch size per device during training
      per_device_eval_batch_size=2,   # batch size for evaluation
      warmup_steps=5000,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      save_total_limit=5,
      logging_steps=10,
      evaluation_strategy='steps',
      eval_steps=500,
      load_best_model_at_end=True,
      gradient_accumulation_steps=128,
      ignore_data_skip=True,
      fp16=True,
  )

  model.parallelize()
  #model.copyweights(pretrained)

  trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=test_dataset,            # evaluation dataset
  )

  try:
      trainer.train(checkpoint)
      trainer.save_model(os.path.join(training_args.output_dir, 'best'))
  except KeyboardInterrupt:
      trainer.save_model(os.path.join(training_args.output_dir, 'interrupt'))
  except Exception as e:
      traceback.print_exc()
      trainer.save_model(os.path.join(training_args.output_dir, 'crash'))

if __name__ == "__main__":
    main()