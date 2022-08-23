from datasets import load_dataset, load_metric, ClassLabel
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import numpy as np
import logging
import copy
import pandas as pd
import yaml

from cls_eval import validate
from utils import AverageMeter


# Set log level
logging.basicConfig(
    level=logging.INFO,
    format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')


def train_one_epoch(dataloader_train, model, optimizer, lr_scheduler, train_epoch_metric, device):

    model.train()
    train_epoch_loss = AverageMeter()
    for batch in dataloader_train:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(**batch)
            logits = outputs.logits
            train_batch_loss = outputs.loss
            predictions = torch.argmax(logits, dim=-1)

            train_batch_loss.backward()
            optimizer.step()

        train_epoch_loss.update(train_batch_loss, batch['input_ids'].size(0))
        train_epoch_metric.add_batch(predictions=predictions, references=batch["labels"])

        lr_scheduler.step()

    return model, train_epoch_loss.avg, train_epoch_metric.compute()['accuracy']


def train(
    dataloaders, model, optimizer, lr_scheduler, num_epochs, metric,
    label_encoder, device, saved_model_path):

    logging.info("Training starts...")
    best_state_dict = copy.deepcopy(model.state_dict())
    best_metric = 0
    train_loss, train_metric, test_loss, test_metric, lr = [], [], [], [], []
    for epoch in range(num_epochs):

        # Train
        model, train_epoch_loss, train_epoch_metric =\
            train_one_epoch(
                dataloaders['train'], model, optimizer, lr_scheduler,
                metric, device)
        train_loss.append(train_epoch_loss.item())
        train_metric.append(train_epoch_metric)
        logging.info(
            f"Epoch {epoch:3d}/{num_epochs-1:3d} {'Train':5s}, "
            f"Loss: {train_epoch_loss:.4f}, "
            f"accuracy: {train_epoch_metric:.4f}")

        # Eval
        test_epoch_loss, test_epoch_metric = validate(
            dataloaders['test'], model, metric, label_encoder, device, False)
        test_loss.append(test_epoch_loss.item())
        test_metric.append(test_epoch_metric)
        logging.info(
            f"Epoch {epoch:3d}/{num_epochs-1:3d} {'Val':5s}, "
            f"Loss: {test_epoch_loss:.4f}, "
            f"accuracy: {test_epoch_metric:.4f}")

        lr.append(lr_scheduler.get_last_lr()[0])

        if test_epoch_metric > best_metric:
            best_metric = test_epoch_metric
            best_state_dict = copy.deepcopy(model.state_dict())

    logging.info(f"Best Val Accuracy: {best_metric:.4f}")

    # Load best model
    model.load_state_dict(best_state_dict)

    # Classification report
    test_epoch_loss, test_epoch_metric = validate(
        dataloaders['test'], model, metric, label_encoder, device, True)

    # Save best model
    best_state_dict = copy.deepcopy(model.state_dict())
    torch.save(best_state_dict, f"{saved_model_path}.pth")

    pd.DataFrame({
        'Epochs': range(num_epochs), 'Learning Rate': lr,
        'Training Loss': train_loss, 'Training Accuracy': train_metric,
        'Validation Loss': test_loss, 'Validation Accuracy': test_metric
        }).to_csv(f"{saved_model_path}.csv", index=False)
    logging.info("Training run successfully...")
    return model


if __name__ == "__main__":
    # Input
    text_column = 'pre_processed'
    label_column = 'label'
    model_name = 'bert-base-cased'  # what are other models
    batch_size = 64
    num_epochs = 1
    dataset_path = 'data/Crime_News_multiclass.csv'
    device = torch.device("cuda:0") if not torch.cuda.is_available() else torch.device("cpu")
    saved_model_path = 'model0'
    # cross entropy loss is used by default

    # Prepare dataset
    dataset = load_dataset(dataset_path.split('.')[-1], data_files=dataset_path)['train']
    unwanted_column = set(dataset.features.keys()).difference(set((text_column, label_column)))
    dataset = dataset.remove_columns(unwanted_column)
    dataset = dataset.rename_column(label_column, "labels")  # label_column must be named as 'labels'

    # Get classes info
    class_names = np.unique(dataset['labels'])
    num_classes = len(class_names)
    label_encoder = ClassLabel(num_classes, class_names)  # how to save label encoder

    # Train-test split
    dataset = dataset.train_test_split(test_size=0.3)

    # Tokenization and label encoding
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(rows):
        return tokenizer(rows['labels'], padding="max_length", truncation=True)
    def convert_name_to_int(rows):
        rows['labels'] = label_encoder.str2int(rows['labels'])
        return rows
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.map(convert_name_to_int, batched=True)
    # tokenized_datasets['train'][0]

    # Remove text column
    tokenized_datasets = tokenized_datasets.remove_columns(text_column)

    # Convert to torch dataset
    tokenized_datasets.set_format("torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    # tokenized_datasets.reset_format()  # to return python object

    # Get train and test dataset
    train_dataset = tokenized_datasets['train']
    test_dataset = tokenized_datasets['test']

    # Training setting
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    dataloaders = {
        'train': train_dataloader, 'test': test_dataloader}
    # next(iter(train_dataloader))

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Metric
    metric = load_metric("accuracy")

    # Save training details
    config = dict(
        text_column=text_column,
        label_column=label_column,
        model_name=model_name,
        dataset_path=dataset_path,
        num_classes=num_classes)
    with open(f"{saved_model_path}.yml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    with open(f'{saved_model_path}_label.txt', 'w') as f:
        f.writelines('\n'.join(list(label_encoder.names)))

    # Train
    model = train(
        dataloaders, model, optimizer, lr_scheduler, num_epochs, metric,
        label_encoder, device, saved_model_path)
