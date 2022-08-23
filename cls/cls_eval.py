import os
import sys
import torch
from torch import nn
from torchvision import transforms
from sklearn.metrics import classification_report
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from utils import AverageMeter


def validate(dataloader_test, model, test_epoch_metric, label_encoder, device, print_cls_report=False, ):

    model.eval()
    test_epoch_loss = AverageMeter()
    labels_all = []
    preds_all = []

    for batch in dataloader_test:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.set_grad_enabled(False):
            outputs = model(**batch)
            logits = outputs.logits
            test_batch_loss = outputs.loss
            predictions = torch.argmax(logits, dim=-1)

            test_epoch_loss.update(test_batch_loss, batch['input_ids'].size(0))
            test_epoch_metric.add_batch(predictions=predictions, references=batch["labels"])

        labels_all.append(batch['labels'])
        preds_all.append(predictions)
    
    labels_all = torch.concat(labels_all, 0).cpu().numpy()
    preds_all = torch.concat(preds_all, 0).cpu().numpy()
    if print_cls_report:
        cls_report = classification_report(
            y_true=labels_all, y_pred=preds_all,
            target_names=label_encoder.names,
            digits=6)
        logging.info(f"\n{cls_report}")

    return test_epoch_loss.avg, test_epoch_metric.compute()['accuracy']


if __name__ == "__main__":
    from datasets import load_dataset, load_metric, ClassLabel
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader

    # Set log level
    logging.basicConfig(
        level=logging.DEBUG,
        format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')

    # Input
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    saved_model_path = "model0"

    # Load config
    with open(f"{saved_model_path}.yml", "r") as stream:
        config = yaml.safe_load(stream)

    # Prepare dataset
    dataset = load_dataset(config['dataset_path'].split('.')[-1], data_files=config['dataset_path'])['train']
    unwanted_column = set(dataset.features.keys()).difference(set((config['text_column'], config['label_column'])))
    dataset = dataset.remove_columns(unwanted_column)
    dataset = dataset.rename_column(config['label_column'], "labels")  # label_column must be named as 'labels'

    # Get classes info
    with open(f'{saved_model_path}_label.txt', 'r') as f:
        class_names = f.readlines()
    class_names = [i.strip() for i in class_names]
    num_classes = config['num_classes']
    label_encoder = ClassLabel(num_classes, class_names)

    # Tokenization and label encoding
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    def tokenize_function(rows):
        return tokenizer(rows['labels'], padding="max_length", truncation=True)
    def convert_name_to_int(rows):
        rows['labels'] = label_encoder.str2int(rows['labels'])
        return rows
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.map(convert_name_to_int, batched=True)
    # tokenized_datasets['train'][0]

    # Remove text column
    tokenized_datasets = tokenized_datasets.remove_columns(config['text_column'])

    # Convert to torch dataset
    tokenized_datasets.set_format("torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    # tokenized_datasets.reset_format()  # to return python object

    # test dataset
    test_dataset = tokenized_datasets

    # Training setting
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    dataloaders = {'test': test_dataloader}

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'], num_labels=num_classes)
    state_dict = torch.load(f'{saved_model_path}.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)

    # Metric
    metric = load_metric("accuracy")

    # eval
    test_epoch_loss, test_epoch_metric = validate(
        dataloaders['test'], model, metric, label_encoder, device, True)
