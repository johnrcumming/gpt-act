import os
import sys
import traceback
from argparse import ArgumentParser
import torch
from tqdm import tqdm

import transformers
import datasets
import wandb

from transformers import GPT2Config
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from transformers import Trainer, TrainingArguments



from gpt2_act import GPT2ACTLMHeadModel, GPT2ACTConfig, GPT2ACTDistilation

def GroupTexts(block_size, tokenizer=None, field='text'):
    def group_texts_fn(examples):
        if tokenizer is not None:
            examples = tokenizer(examples[field])
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys() }
        #print('concatenated_examples', {k: len(concatenated_examples[k]) for k in concatenated_examples.keys() })

        total_length = len(concatenated_examples[list(examples.keys())[0]])
        #print('total_length', total_length)

        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        #print('total_length', total_length)

        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        #print('result', {k: len(result[k]) for k in result.keys() })

        result["labels"] = result["input_ids"].copy()
        #print('result', {k: len(result[k][0] if isinstance(result[k][0], list) else result[k]) for k in result.keys() })

        return result
    return group_texts_fn

def load_streaming_dataset(model_config, data_dir='data', dataset_name='wikitext', dataset_config=None, num_procs=10, val_size=0.05, test_size=0.05, group_texts=True):
    """Load a streaming dataset from the datasets library, and preprocess it for training."""
    dataset_dir = os.path.join(data_dir, dataset_name)
    cache_dir = os.path.join(data_dir, 'cache')

    if dataset_config is None:
        dataset_config = datasets.get_dataset_config_names(dataset_name)[0]

    tokenizer = GPT2Tokenizer.from_pretrained(model_config)
    tokenizer.pad_token = tokenizer.eos_token
    model_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e20)  # LARGE_INTEGER

    dataset = datasets.load_dataset(dataset_name, dataset_config, data_dir=dataset_dir, cache_dir=cache_dir, streaming=True)

    if 'text' in dataset['train'].column_names:
        field = 'text'
    elif 'sentence' in dataset['train'].column_names:
        field = 'sentence'
    elif 'content' in dataset['train'].column_names:
        field = 'content'
    else:
        raise ValueError('Dataset does not contain a text field.')
    
    if group_texts:
        dataset = dataset.map(GroupTexts(model_max_length, tokenizer=tokenizer, field=field), batched=True, remove_columns=dataset['train'].column_names)

    return dataset

def preprocess_dataset(model_config, data_dir='data', cache_dir=None, dataset_name='wikitext', dataset_config=None, num_procs=10, val_size=0.05, test_size=0.05, group_texts=True):
    """Load a dataset from the datasets library, and preprocess it for training. and save dataset locally."""
    dataset_dir = os.path.join(data_dir, dataset_name)
    if cache_dir is None:
        cache_dir = os.path.join(data_dir, 'cache')

    if not os.path.exists(dataset_dir):
        if dataset_config is None:
            dataset_config = datasets.get_dataset_config_names(dataset_name)[0]

        tokenizer = GPT2Tokenizer.from_pretrained(model_config)
        tokenizer.pad_token = tokenizer.eos_token
        model_max_length = tokenizer.model_max_length
        tokenizer.model_max_length = int(1e20)  # LARGE_INTEGER

        dataset = datasets.load_dataset(dataset_name, dataset_config, data_dir=dataset_dir, cache_dir=cache_dir, num_proc=num_procs)

        if 'text' in dataset['train'].column_names:
            field = 'text'
        elif 'sentence' in dataset['train'].column_names:
            field = 'sentence'
        elif 'content' in dataset['train'].column_names:
            field = 'content'
        else:
            raise ValueError('Dataset does not contain a text field.')
        
        if group_texts:
            dataset = dataset.map(GroupTexts(model_max_length, tokenizer=tokenizer, field=field), num_proc=num_procs, batched=True, remove_columns=dataset['train'].column_names)

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
          model_config='gpt2', checkpoint=None, verbose=True, fp16=False, 
          pretrained=None, freeze_pretrained=False, lambda_kd=1e-4, temperature_kd=4.0, max_grad_norm=1.0,
          stream_dataset=False, max_steps=-1, storage_options=None, num_procs=10,
          push_to_hub_model_id=None, push_to_hub_organization=None, push_to_hub_token=None,
          report_to="all", run_name=None, no_cuda=False, logging_steps=10, save_steps=500, eval_steps=0, warmup_steps=5000, learning_rate=1e-5,
          deepspeed_config=None, dynamic_stride=None, distill=False,
          binary_embedding=False, n_positions=1024, halting_function_spec=None, layerwise_attn="simple", group_texts=True, act_depth=None):    
    """Train a GPT2ACT model on a dataset."""

    wandb.init(project='gpt2act', name=run_name)

    if verbose:
        transformers.utils.logging.set_verbosity_info()

    transformers.set_seed(42)

    if stream_dataset:
        dataset = load_streaming_dataset(model_config, data_dir=data_dir, dataset_name=dataset_name, group_texts=group_texts)
        train_dataset = dataset['train']
        val_dataset = dataset['validation'] if 'validation' in dataset else None

    else:
        dataset_dir=os.path.join(data_dir, dataset_name)
        dataset = datasets.DatasetDict.load_from_disk(dataset_dir)
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        
    gpt2_config = GPT2Config.from_pretrained(model_config)
    gpt2_config.n_positions=n_positions
    gpt2_config.tie_word_embeddings = not binary_embedding

    config = GPT2ACTConfig(act_commitment_cost=act_commitment_cost,
                           gradient_checkpointing=gradient_checkpointing,
                           dynamic_stride=dynamic_stride,
                           lambda_kd=lambda_kd, temperature_kd=temperature_kd,
                           teacher=model_config,
                           use_binary_embedding=binary_embedding,
                           halting_function_spec=halting_function_spec,
                           layerwise_attn=layerwise_attn,
                           act_depth=act_depth,
                           **gpt2_config.to_dict())
    
    if distill:
        model = GPT2ACTDistilation(config)
    else:
        model = GPT2ACTLMHeadModel(config)

    if pretrained:
        model.copyweights(model_config, freeze_pretrained)

    os.makedirs(base_logging_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if run_name is None:
        run_dir = dataset_name + str(len(os.listdir(base_logging_dir)))
        logging_dir = os.path.join(base_logging_dir, run_dir)
        checkpoint_dir = os.path.join(checkpoint_dir, run_dir)
    else:
        logging_dir = os.path.join(base_logging_dir, run_name)
        checkpoint_dir = os.path.join(checkpoint_dir, run_name)

    if eval_batch_size > 0:
        if eval_steps > 0:
            evaluation_strategy = 'steps'
        else:
            evaluation_strategy = 'epoch'
    else:
        evaluation_strategy = 'no'

    training_args = TrainingArguments(
        output_dir= checkpoint_dir,          # output directory
        logging_dir= logging_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,              # total # of training epochs
        per_device_train_batch_size=train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=eval_batch_size if eval_batch_size > 0 else None,   # batch size for evaluation
        warmup_steps=warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        save_total_limit=5,
        logging_steps=logging_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy='steps',
        save_steps=save_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_accumulation_steps=gradient_accumulation_steps,
        ignore_data_skip=True,
        fp16=fp16,
        max_steps=max_steps,
        dataloader_pin_memory=True,
        do_train=True,
        do_eval=eval_batch_size > 0,
        dataloader_num_workers=num_procs,
        push_to_hub_model_id=push_to_hub_model_id,
        push_to_hub_organization=push_to_hub_organization,
        push_to_hub_token=push_to_hub_token,
        report_to=report_to,
        run_name=run_name,
        no_cuda=no_cuda,
        learning_rate=learning_rate,
        deepspeed=deepspeed_config,
        max_grad_norm=max_grad_norm,

    )

    if parallelize:
        model.parallelize()
    
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

def calculate_perplexity(data_dir='data', dataset_name='wikitext', model_config='gpt2', checkpoint=None, num_procs=4, verbose=True, no_cuda=True, fp16=True):
    if verbose:
        transformers.utils.logging.set_verbosity_info()

    model = GPT2ACTLMHeadModel.from_pretrained(checkpoint, torch_dtype=torch.float16 if fp16 else torch.float).to('cpu' if no_cuda else 'cuda')

    dataset_dir=os.path.join(data_dir, dataset_name)
    dataset = datasets.DatasetDict.load_from_disk(dataset_dir)
    encodings = dataset['test']

    max_length = model.config.n_positions
    nlls = []
    prev_end_loc = 0

    for batch in tqdm(encodings):
        input_ids = torch.tensor(batch['input_ids']).unsqueeze(0).to('cpu' if no_cuda else 'cuda')
        target_ids = input_ids.clone()
        target_ids[:, :-1] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).mean())

    print('Perplexity:', float(ppl))


def main():
    parser = ArgumentParser(description='Perform GPT2ACT Dataset Build and Training.', add_help=False)

    parser.add_argument('--preprocess_dataset', default=False, action='store_true', help='Build Dataset.')
    parser.add_argument('--train_model', default=False, action='store_true', help='Train Model.')
    parser.add_argument('--calculate_perplexity', default=False, action='store_true', help='Calculate Perplexity of Trained Model.')


    parser.add_argument('--model_config', type=str, default="gpt2", help='Huggingface Transformers Model Config.')
    parser.add_argument('--dataset_name', type=str, default="wikitext", help='Huggingface Datasets Name.')
    parser.add_argument('--dataset_config', type=str, default=None, help='Huggingface Datasets Configuration Name.')

    parser.add_argument('--deepspeed_config', type=str, default=None, help='Deepspeed JSON Config File.')
    parser.add_argument('--data_dir', type=str, default='data', help='Dataset Directory.')
    parser.add_argument('--log_dir', type=str, default='runs', help='Tensorboard Log Dir.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Ceckpoint Save Dir.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to Continue From.')
    parser.add_argument('--distill', default=False,  action='store_true', help='Distill Model from Pretrained.')
    parser.add_argument('--pretrained', default=False,  action='store_true', help='Copy Weights from GPT2 model_config.')
    parser.add_argument('--freeze_pretrained', default=False,  action='store_true', help='Freeze pretrained weights Training.')
    parser.add_argument('--lambda_kd', type=float, default=1e-6, help='Knowledge Distillation Loss Weight.')
    parser.add_argument('--temperature_kd', type=float, default=4.0, help='Knowledge Distillation temperature_kd.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient Clipping Max Grad Norm.')

    parser.add_argument('--dynamic_stride', type=int, default=None, help='Dynamic Block Stride.')

    parser.add_argument('--num_procs', type=int, default=2, help='Number of Processes for Dataset Processing.')
    parser.add_argument('--logging_steps', type=int, default=10, help='Log every n steps')
    parser.add_argument('--save_steps', type=int, default=500, help='Save checkpoint every n steps.')
    parser.add_argument('--eval_steps', type=int, default=0, help='Evaluate every n steps, default evaluate at epoch.')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Optimizer Warmup steps.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Optimizer Learning Rate.')

    parser.add_argument('--binary_embedding', default=False, action='store_true', help='Use Experimental Binary Embedding.')
    parser.add_argument('--n_positions', type=int, default=1024, help='n_positions - context length.')

    parser.add_argument('--train_epochs', type=int, default=5, help='Training Epochs.')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Training Batch Size.')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Evaluation Batch Size.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32, help='Gradient Accumulation Steps.')
    parser.add_argument('--parallelize', default=False, action='store_true', help='Parallelize Model - split model across GPUs.')
    parser.add_argument('--verbose', default=False, action='store_true', help='Verbose Logging.')
    parser.add_argument('--stream_dataset', default=False, action='store_true', help='Stream Dataset.')
    parser.add_argument('--no_fp16', dest='fp16', default=True, action='store_false', help='FP16 Training.')
    parser.add_argument('--max_steps', type=int, default=-1, help='Number of train steps for streaming_datasets.')
    parser.add_argument('--act_commitment_cost', type=float, default=1e-4, help='ACT Loss commitmemt cost.')
    parser.add_argument('--gradient_checkpointing', default=False, action='store_true', help='Enable Gradient Chackpointing.')
    parser.add_argument('--no_cuda', default=False, action='store_true', help='Disable CUDA.')

    parser.add_argument('--push_to_hub_model_id', type=str, default=None, help='The name of the repository to which push the Trainer.')
    parser.add_argument('--push_to_hub_organization', type=str, default=None, help='The name of the organization in with to which push the Trainer.')
    parser.add_argument('--push_to_hub_token', type=str, default=None, help=' The token to use to push the model to the Hub.')

    parser.add_argument('--report_to', type=str, default="all", help='The list of integrations to report the results and logs to. Supported platforms are "azure_ml", "comet_ml", "mlflow", "tensorboard" and "wandb". Use "all" to report to all integrations installed, "none" for no integrations.')
    parser.add_argument('--run_name', type=str, default=None, help='A descriptor for the run. Typically used for wandb logging.')

    parser.add_argument('--halting_function_spec', type=str, default=None, help='Halting Function Spec.')
    parser.add_argument('--layerwise_attn', type=str, default='simple', choices=['simple', 'mha', 'sha'], help='Set Layerwise Attention.')

    parser.add_argument('--no_group_texts', dest='group_texts', default=True, action='store_false', help='Disable Text Grouping')

    parser.add_argument('--act_depth_factor', type=float, default=None, help='ACT Depth Factor.')
    parser.add_argument('--act_depth', type=int, default=None, help='ACT Depth.')


    args = parser.parse_args()
    
    if args.act_depth_factor is not None:
        act_depth = args.act_depth_factor
    elif args.act_depth is not None:
        act_depth = args.act_depth
    else:
        act_depth = None

    if args.preprocess_dataset:
        preprocess_dataset(args.model_config, data_dir=args.data_dir, 
                           dataset_name=args.dataset_name, dataset_config=args.dataset_config, num_procs=args.num_procs, group_texts=args.group_texts)

    if args.train_model:
        train(args.data_dir, args.log_dir, args.checkpoint_dir, args.dataset_name,
                num_train_epochs=args.train_epochs, train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps, parallelize=args.parallelize,
                gradient_checkpointing=args.gradient_checkpointing, act_commitment_cost=args.act_commitment_cost,
                model_config=args.model_config, checkpoint=args.checkpoint, 
                pretrained=args.pretrained, freeze_pretrained=args.freeze_pretrained, lambda_kd=args.lambda_kd, temperature_kd=args.temperature_kd,
                verbose=args.verbose, stream_dataset=args.stream_dataset, fp16=args.fp16, max_steps=args.max_steps, num_procs=args.num_procs,
                push_to_hub_model_id=args.push_to_hub_model_id, push_to_hub_organization=args.push_to_hub_organization, push_to_hub_token=args.push_to_hub_token,
                report_to=args.report_to, run_name=args.run_name,
                no_cuda=args.no_cuda, logging_steps=args.logging_steps, save_steps=args.save_steps, eval_steps=args.eval_steps, learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps, deepspeed_config=args.deepspeed_config, dynamic_stride=args.dynamic_stride,
                max_grad_norm=args.max_grad_norm, distill=args.distill, group_texts=args.group_texts, 
                binary_embedding=args.binary_embedding, n_positions=args.n_positions, halting_function_spec=args.halting_function_spec, layerwise_attn=args.layerwise_attn,
                act_depth = act_depth
             )
        
    if args.calculate_perplexity:
        calculate_perplexity(data_dir=args.data_dir, dataset_name=args.dataset_name, checkpoint=args.checkpoint, 
                             num_procs=args.num_procs, verbose=args.verbose, no_cuda=args.no_cuda, fp16=args.fp16)
if __name__ == "__main__":
    main()
