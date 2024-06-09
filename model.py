import numpy as np
import evaluate
from Lib.config_loader import config
import torch

from transformers import (
    Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification
)

from datasets import Dataset
from datasets.dataset_dict import DatasetDict

train = config["model"]["train"]
val = config["model"]["val"]
test = config["model"]["test"]

model_checkpoint = config["model"]["model_checkpoint"]
metric_name = config["model"]["metric_name"]

batch_size = config["model"]["batch_size"]
epoch_num = config["model"]["epoch_num"]

class Model:

    def __init__(self, data, labels, unique_label_count):

        self.total_size = len(data)

        #model_checkpoint = "dccuchile/bert-base-spanish-wwm-uncased"
        self.model = BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=unique_label_count)
        self.model_name = model_checkpoint.split("/")[-1]

        train_set = Dataset.from_dict({"data": data[:int(self.total_size*train)], "labels": labels[:int(self.total_size*train)]})
        val_set = Dataset.from_dict({"data": data[int(self.total_size*train):][:int(self.total_size*val)], "labels": labels[int(self.total_size*train):][:int(self.total_size*val)]})
        test_set = Dataset.from_dict({"data": data[-int(self.total_size*test):], "labels": labels[int(-self.total_size*test):]})
        full_dataset = DatasetDict({"train": train_set, "validation": val_set, "test": test_set})

        self.metric = evaluate.load(metric_name)

        self.tokenizer = BertTokenizer.from_pretrained(model_checkpoint, do_lower_case=False)
        self.tokenized_dataset = full_dataset.map(self.tokenize_fn, batched=True, batch_size=32)
        self.tokenized_dataset.set_format("torch")

        # freeze todas las capas
        for param in self.model.parameters():
            param.requires_grad = False

        # descongelar las ultimas capas
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        # y el ultimo transformer block:
        for param in self.model.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

        args = TrainingArguments(
            f"{self.model_name}-finetuned-cola",
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epoch_num,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            push_to_hub=False,
            seed=33,
        )

        self.trainer = Trainer(
            self.model,
            args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        self.results = self.trainer.train()
        self.logits = zip(self.test_logits(), test_set["labels"])

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=labels)

    def tokenize_fn(self, examples):
        return self.tokenizer(examples["data"], truncation=True, padding=True)
    
    def test_logits(self):
        inputs = self.tokenizer(self.tokenized_dataset["test"]["data"], return_tensors='pt', padding=True, truncation=True)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits
