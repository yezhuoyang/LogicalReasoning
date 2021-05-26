'''
Load a model from pytorch_model.bin and make test prediction
'''
import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from utils_multiple_choice import convert_examples_to_features, processors



from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMultipleChoice,
    BertTokenizer,
    RobertaConfig,
    RobertaForMultipleChoice,
    RobertaTokenizer,
    XLNetConfig,
    XLNetForMultipleChoice,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

logger=logging.getLogger(__name__)




def evaluate(model, tokenizer, prefix="", test=False):
    eval_task_names = ("reclor",)
    eval_outputs_dirs = ("/Users/yezhuoyang/Desktop/LogicalReasoning(git)/reclor",)
    device=torch.device("cpu")
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(eval_task, tokenizer, evaluate=not test, test=test)


        eval_batch_size =8*max(1,0)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()


            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        acc = simple_accuracy(preds, out_label_ids)

        result = {"eval_acc": acc, "eval_loss": eval_loss}
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "is_test_" + str(test).lower() + "_eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(str(prefix) + " is test:" + str(test)))
            if not test:
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
    if test:
        return results, preds
    else:
        return results



def load_and_cache_examples(task, tokenizer, evaluate=False, test=False):
    data_dir="/Users/yezhuoyang/Desktop/LogicalReasoning(git)/reclor/reclor_data"
    processor = processors['reclor']()
    max_seq_length=256
    model_name_or_path="bert-base-uncased"
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = "dev"
    elif test:
        cached_mode = "test"
    else:
        cached_mode = "train"
    assert not (evaluate and test)
    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}_{}".format(
            cached_mode,
            list(filter(None, model_name_or_path.split("/"))).pop(),
            str(max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(data_dir)
        elif test:
            examples = processor.get_test_examples(data_dir)
        else:
            examples = processor.get_train_examples(data_dir)
        logger.info("Training number: %s", str(len(examples)))
        features = convert_examples_to_features(
            examples,
            label_list,
            max_seq_length,
            tokenizer,
            pad_on_left=False,  # pad on the left for xlnet
            pad_token_segment_id=0,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)

    all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset







'''
Make a prediction from given checkpoint. 
'''
if __name__ == "__main__":
    output_dir="/Users/yezhuoyang/Desktop/LogicalReasoning(git)/reclor"
    config_class=BertConfig
    tokenizer_class=BertTokenizer
    model_class=BertForMultipleChoice

    checkpoint_dir = "/Users/yezhuoyang/Desktop/LogicalReasoning(git)/reclor/Checkpoints/reclor/bert-base-uncased/checkpoint-1750"

    model=model_class.from_pretrained(checkpoint_dir)

    tokenizer=tokenizer_class.from_pretrained(checkpoint_dir)
    result, preds=evaluate(model, tokenizer, prefix="", test=True)

    np.save(os.path.join(output_dir, "test_preds.npy" if output_dir is not None else "test_preds.npy"), preds)