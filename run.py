import argparse
import logging
import os
import random
import torch
import json
import numpy as np
from model import Model
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaModel, RobertaTokenizer)
from dataloader import TextDataset, textdataset_collate_fn, textdataset_noalign_collate_fn, normalize_token
from typing import Optional
logger = logging.getLogger(__name__)
import multiprocessing
from multiprocessing import Pool
from typing import List
import wandb # See https://docs.wandb.ai/quickstart/
import matplotlib.pyplot as plt
import datetime
cpu_count = 16
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable synchronous CUDA execution for debugging

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def lambda_lr(epoch):
    if epoch == 0:
        return 0.0  # Learning rate for the first epoch is 0
    else:
        return 1.0  # Subsequent epochs use args.learning_rate


def train(args: argparse.Namespace, model: Model, tokenizer: RobertaTokenizer, pool: Pool):
    """ Train the model """
    # Load labeled samples first
    full_dataset = TextDataset(tokenizer, args, args.train_data_file, pool)

    # Load roles data
    logger.info("Loading roles data from %s", args.roles_file)
    roles_data = np.load(args.roles_file)
    roles = torch.from_numpy(roles_data).long().to(args.device)
    logger.info("Loaded roles data with shape: %s", roles.shape)

    # If training on a single sample, create a custom dataset with just that sample
    if args.train_single_sample:
        if args.target_sample_index >= len(full_dataset):
            logger.error(f"Target sample index {args.target_sample_index} is out of range (dataset size: {len(full_dataset)})")
            return
        
        logger.info(f"Training on single sample at index {args.target_sample_index}")
        # Get the single sample
        single_sample = full_dataset[args.target_sample_index]
        # Create a custom dataset with just this sample
        class SingleSampleDataset(torch.utils.data.Dataset):
            def __init__(self, sample):
                self.sample = sample
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return self.sample
        
        train_dataset = SingleSampleDataset(single_sample)
        logger.info("Created single sample dataset")
    else:
        train_dataset = full_dataset

    # Create dataloader
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=1 if args.train_single_sample else args.train_batch_size,
                                  num_workers=4,
                                  collate_fn=textdataset_collate_fn)

    # Get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * args.num_train_epochs)

    # Multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Load checkpoint if resuming from a specific epoch
    start_epoch = 0
    resume_epoch: Optional[int] = None
    if resume_epoch is not None:
        checkpoint_path = os.path.join(args.output_dir, f'Epoch_{resume_epoch}', 'subject_model.pth')
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from epoch {resume_epoch}, path {checkpoint_path}")
            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(torch.load(checkpoint_path), strict=False)
            else:
                model.load_state_dict(torch.load(checkpoint_path), strict=False)
            start_epoch = resume_epoch
        else:
            logger.warning(f"Checkpoint for epoch {resume_epoch} not found. Starting from beginning.")

    # Train!
    logger.info("***** Running training *****")
    if args.train_single_sample:
        logger.info("  Training on single sample at index %d", args.target_sample_index)
    else:
        logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", 1 if args.train_single_sample else args.train_batch_size // args.n_gpu)
    logger.info("  Total train batch size = %d", 1 if args.train_single_sample else args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader) * args.num_train_epochs)
    if resume_epoch:
        logger.info("  Resuming from epoch %d", resume_epoch)

    model.zero_grad()
    model.train()
    # model.set_tokenizer(tokenizer)

    accumulation_steps = 8  # Set to 8, so that each epoch updates parameters about 20-25 times, balancing training stability and efficiency
    optimizer.zero_grad()

    # Set the epoch to start adding retrieval loss
    retrieval_start_epoch = args.num_train_epochs + 1 # no retrieval at all

    for idx in range(start_epoch, args.num_train_epochs):
        total_epoch_attention_loss = 0
        total_epoch_retrieval_loss = 0
        total_epoch_cross_sample_loss = 0
        total_batches = 0

        for step, batch in enumerate(train_dataloader):
            total_batches += 1

            # Get inputs
            code_inputs = batch[0].to(args.device)
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            nl_inputs = batch[3].to(args.device)
            ori2cur_pos = batch[4].to(args.device)
            match_list = batch[5]
            valid_code_spans_batch = batch[6]  # Valid code spans for each sample
            valid_comment_spans_batch = batch[7]  # Valid comment spans for each sample

            # Get batch indices for roles
            if args.train_single_sample:
                batch_roles = roles[args.target_sample_index:args.target_sample_index+1]
            else:
                batch_start_idx = step * args.train_batch_size
                batch_end_idx = min((step + 1) * args.train_batch_size, len(full_dataset))
                batch_roles = roles[batch_start_idx:batch_end_idx]

            code_outputs, nl_outputs = model(
                code_inputs=code_inputs,
                attn_mask=attn_mask,
                position_idx=position_idx,
                nl_inputs=nl_inputs,
                role_indices=batch_roles  # Pass only the roles for current batch
            )

            total_attention_loss = torch.tensor(0.0, device=code_inputs.device)
            total_nl_highlight_loss = torch.tensor(0.0, device=code_inputs.device)
            total_code_highlight_loss = torch.tensor(0.0, device=code_inputs.device)
            total_cross_sample_loss = torch.tensor(0.0, device=code_inputs.device)

            # Process each sample in batch
            for batch_idx in range(code_inputs.size(0)):
                code_tokens_2 = (code_inputs[batch_idx] == 2).nonzero().flatten()
                nl_tokens_2 = (nl_inputs[batch_idx] == 2).nonzero().flatten()

                # Set default values if no token 2 found
                total_code_tokens = min(ori2cur_pos[batch_idx].max().item(), 255)
                if len(code_tokens_2) == 0:
                    total_code_tokens = 255 - 1
                else:
                    total_code_tokens = int(code_tokens_2[0].item())

                if len(nl_tokens_2) == 0:
                    total_comment_tokens = 127 - 1
                else:
                    total_comment_tokens = int(nl_tokens_2[0].item())

                if isinstance(model, torch.nn.DataParallel):
                    total_highlight_loss, nl_highlight_loss, code_highlight_loss = model.module.compute_loss(
                        code_inputs, code_outputs, nl_outputs, batch_idx, match_list[batch_idx],
                        total_code_tokens, total_comment_tokens, batch_roles, valid_code_spans_batch[batch_idx]
                    )
                else:
                    total_highlight_loss, nl_highlight_loss, code_highlight_loss = model.compute_loss(
                        code_inputs, code_outputs, nl_outputs, batch_idx, match_list[batch_idx],
                        total_code_tokens, total_comment_tokens, batch_roles, valid_code_spans_batch[batch_idx]
                    )

                total_attention_loss += total_highlight_loss
                total_nl_highlight_loss += nl_highlight_loss
                total_code_highlight_loss += code_highlight_loss

            # Compute cross-sample contrastive loss with filtered negative samples
            if len(valid_code_spans_batch) > 1:  # Only compute if we have multiple samples
                if isinstance(model, torch.nn.DataParallel):
                    cross_sample_loss = model.module.compute_cross_sample_contrastive_loss_with_filtering(
                        nl_outputs.last_hidden_state,
                        code_outputs.last_hidden_state,
                        match_list,
                        [min(ori2cur_pos[b].max().item(), 255) for b in range(code_inputs.size(0))],
                        [127 - 1 if len((nl_inputs[b] == 2).nonzero().flatten()) == 0 else int((nl_inputs[b] == 2).nonzero().flatten()[0].item()) for b in range(code_inputs.size(0))],
                        valid_comment_spans_batch,
                        valid_code_spans_batch,
                        similarity_threshold=0.3,
                        max_negative_samples_per_concept=50,  # Limit to 50 negative samples per concept
                        code_inputs=code_inputs,
                        nl_inputs=nl_inputs
                    )
                else:
                    cross_sample_loss = model.compute_cross_sample_contrastive_loss_with_filtering(
                        nl_outputs.last_hidden_state,
                        code_outputs.last_hidden_state,
                        match_list,
                        [min(ori2cur_pos[b].max().item(), 255) for b in range(code_inputs.size(0))],
                        [127 - 1 if len((nl_inputs[b] == 2).nonzero().flatten()) == 0 else int((nl_inputs[b] == 2).nonzero().flatten()[0].item()) for b in range(code_inputs.size(0))],
                        valid_comment_spans_batch,
                        valid_code_spans_batch,
                        similarity_threshold=0.3,
                        max_negative_samples_per_concept=50,  # Limit to 50 negative samples per concept
                        code_inputs=code_inputs,
                        nl_inputs=nl_inputs
                    )
                total_cross_sample_loss = cross_sample_loss

            # Compute retrieval loss (enabled after the specified epoch)
            if idx >= retrieval_start_epoch:
                if isinstance(model, torch.nn.DataParallel):
                    retrieval_loss = model.module.retrieval_loss(
                        code_outputs, nl_outputs
                    )
                else:
                    retrieval_loss = model.retrieval_loss(
                        code_outputs, nl_outputs
                    )
                retrieval_loss = retrieval_loss / code_inputs.size(0)
                total_epoch_retrieval_loss += retrieval_loss.item()
                retrieval_loss.backward() # fixme: retrieval loss only
            else:
                # Use highlight loss + cross-sample loss
                aa_loss = (total_attention_loss) / code_inputs.size(0)
                total_loss = aa_loss + total_cross_sample_loss * 2
                total_loss.backward()

            total_epoch_attention_loss += total_attention_loss.item() / code_inputs.size(0)
            total_epoch_cross_sample_loss += total_cross_sample_loss.item()

            # after computing losses and scheduler step…
            attention_loss = total_attention_loss.item() / code_inputs.size(0)
            cross_sample_loss_val = total_cross_sample_loss.item()
            retr_loss = retrieval_loss.item() if idx >= retrieval_start_epoch else 0.0
            lr = scheduler.get_last_lr()[0]

            if idx >= retrieval_start_epoch:
                logger.info(
                    "train epoch=%d step=%d attention_loss=%.5f cross_sample_loss=%.5f retrieval_loss=%.5f",
                    idx + 1,
                    step + 1,
                    attention_loss,
                    cross_sample_loss_val,
                    retr_loss
                )
            else:
                logger.info(
                    "train epoch=%d step=%d attention_loss=%.5f cross_sample_loss=%.5f",
                    idx + 1,
                    step + 1,
                    attention_loss,
                    cross_sample_loss_val
                )

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Calculate average losses for the epoch
        avg_attention_loss = total_epoch_attention_loss / total_batches
        avg_retrieval_loss = total_epoch_retrieval_loss / total_batches if idx >= retrieval_start_epoch else 0
        avg_cross_sample_loss = total_epoch_cross_sample_loss / total_batches

        if idx >= retrieval_start_epoch:
            logger.info("Epoch {} average losses - Retrieval Loss: {:.5f}, Cross-Sample Loss: {:.5f}".format(
                idx + 1, avg_retrieval_loss, avg_cross_sample_loss
            ))
        else:
            logger.info("Epoch {} average losses - Attention Loss: {:.5f}, Cross-Sample Loss: {:.5f}".format(
                idx + 1, avg_attention_loss, avg_cross_sample_loss
            ))

        # Save checkpoint
        output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(idx + 1))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt_name = 'subject_model_single_sample.pth' if args.train_single_sample else 'subject_model_java.pth'
        ckpt_output_path = os.path.join(output_dir, ckpt_name)
        logger.info("Saving model checkpoint to %s", ckpt_output_path)
        torch.save(model_to_save.state_dict(), ckpt_output_path)

        print("Model saved.")



def evaluate(args, model, tokenizer, file_name, pool, epoch_id, eval_when_training=False):

    query_dataset = TextDataset(tokenizer, args, file_name, pool, compute_alignment=False)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset,
                                  sampler=query_sampler,
                                  batch_size=args.eval_batch_size,
                                  collate_fn=textdataset_noalign_collate_fn,
                                  num_workers=4)

    code_dataset = TextDataset(tokenizer, args, args.codebase_file, pool, compute_alignment=False)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset,
                                 sampler=code_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=textdataset_noalign_collate_fn,
                                 num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    code_vecs = []
    nl_vecs = []
    for batch in query_dataloader:
        nl_inputs = batch[3].to(args.device)

        with torch.no_grad():
            outputs = model(nl_inputs=nl_inputs)
            nl_vec = outputs.nl_hidden.detach().cpu().numpy()
            nl_vecs.extend(nl_vec)

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)

        with torch.no_grad():
            outputs = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
            code_vec = outputs.code_hidden.detach().cpu().numpy()
            code_vecs.extend(code_vec)

    model.train()
    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)

    scores = np.matmul(nl_vecs, code_vecs.T)

    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    nl_urls = []
    code_urls = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)

    for example in code_dataset.examples:
        code_urls.append(example.url)

    ranks = []
    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr": float(np.mean(ranks))
    }
    print(f"[{datetime.datetime.now()}] Epoch {epoch_id + 1} Evaluation - MRR: {result['eval_mrr']:.4f}")
    rank_path = os.path.join(args.output_dir, 'Epoch_{}'.format(epoch_id + 1), 'new_valid_rank.json')
    with open(rank_path, 'w') as file:
        json.dump(ranks, file)

    print(f"Data has been saved to: {rank_path}")

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")
    parser.add_argument("--role_embedding_file", default=None, type=str, required=True,
                        help="Path to the pre-trained role embeddings file (.npy)")
    parser.add_argument("--roles_file", default=None, type=str, required=True,
                        help="Path to the roles data file (.npy)")

    parser.add_argument("--lang", default=None, type=str,
                        help="language.")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--do_calculate_losses", action='store_true', default=False,
                        help="Whether to calculate and save losses per sample.")
    parser.add_argument("--calculate_losses_epoch", default=None, type=int,
                        help="The epoch number of the model checkpoint to load for calculating losses.")

    parser.add_argument("--print_alignment_details", action='store_true', default=False,
                        help="Whether to print detailed alignment loss calculation for a specific sample.")
    parser.add_argument("--target_sample_index", type=int, default=0,
                        help="The index of the sample to print alignment loss details for.")

    # Add new argument for single sample training
    parser.add_argument("--train_single_sample", action='store_true',
                        help="Whether to train on a single sample.")

    # print arguments
    args = parser.parse_args()

    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    pool = multiprocessing.Pool(cpu_count)
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    # build model
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    encoder = RobertaModel.from_pretrained(args.model_name_or_path)
    model = Model(encoder, num_roles=19)  # Initialize with number of roles
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)

    # Training
    if args.do_train:
        train(args, model, tokenizer, pool)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir), strict=False)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.eval_data_file, pool, epoch_id=-1)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir), strict=False)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.test_data_file, pool, epoch_id=-1)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    return results


if __name__ == "__main__":
    main()
