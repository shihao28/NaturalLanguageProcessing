from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed)
import nltk
import numpy as np

from conf import Config


# ==============================================
# FINE-TUNING MODEL ON SUMMARIZATION TASK
# ==============================================
class Train:
    def __init__(self, config):
        self.config = config.TRAIN
        set_seed(self.config["seed"])
        self.model_checkpoint = self.config["model_checkpoint"]        
        # load dataset, tokenizer, metric
        self.raw_datasets = load_dataset(self.config["dataset_name"], self.config["dataset_config_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.metric = load_metric("rouge")

        if self.model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
            self.prefix = "summarize: "
        else:
            self.prefix = ""

        self.max_input_length = self.config["max_source_length"]
        self.max_target_length = self.config["max_target_length"]

        self.batch_size = self.config["batch_size"]
        self.model_name = self.model_checkpoint.split("/")[-1]

    def _preprocess_function(self, examples):
        inputs = [self.prefix + doc for doc in examples[self.config["text_column"]]]
        model_inputs = self.tokenizer(
            inputs, max_length=self.max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples[self.config["summary_column"]],
                max_length=self.max_target_length,
                truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
 
    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels,
            use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def train(self):
        # Setup dataset pipeline
        tokenized_datasets = self.raw_datasets.map(self._preprocess_function, batched=True)

        # Setup model
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=model)

        args = Seq2SeqTrainingArguments(
            f"{self.model_name}-finetuned-{self.config['dataset_name']}",
            evaluation_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=1,
            predict_with_generate=True,
            fp16=False,
            push_to_hub=False,
        )

        # Train
        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics
        )
        train_result = trainer.train()

        # Log best model and metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        print(metrics)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    Train(Config).train()
