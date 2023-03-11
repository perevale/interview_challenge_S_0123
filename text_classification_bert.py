import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from sklearn.model_selection import train_test_split
import torch


def load_dataset(file_path=None):
    if file_path is None:
        file_path = './data/sportoclanky.csv'
    assert os.path.exists(file_path)
    df = pd.read_csv(file_path)
    labelencoder = LabelEncoder()
    df['category_enc'] = labelencoder.fit_transform(df['category'])
    df['text'] = df['rss_title'] + ' ' + df['rss_perex']
    return df


def split_stratified_into_train_val_test(df_input, stratify_colname='y',
                                         frac_train=0.6, frac_val=0.15, frac_test=0.25,
                                         random_state=None):
    
    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test


class Perex_Dataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer):

        self.labels = df['category_enc'].tolist()
        self.texts = tokenizer(df['text'].tolist(), 
                               padding='max_length', max_length = 512, 
                               truncation=True, return_tensors="pt"
                               )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.texts.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item



def main():
    df = load_dataset()
    no_classes = len(df['category_enc'].unique())
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
       "distilbert-base-multilingual-cased", num_labels=no_classes
        )       
    
    f1_metric = evaluate.load("f1")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return f1_metric.compute(predictions=predictions, references=labels, average='weighted')
    
    training_args = TrainingArguments(
        output_dir="bert_classification",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    df_train, df_val, df_test = split_stratified_into_train_val_test(df, stratify_colname='category_enc')
    train, val, test = Perex_Dataset(df_train, tokenizer), Perex_Dataset(df_val, tokenizer), Perex_Dataset(df_test, tokenizer)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print("Evaluation on valid data:")
    print(trainer.evaluate())
    print("Evaluation on test data:")
    print(trainer.predict()[2])


