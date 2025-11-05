import torch
torch.cuda.is_available()

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss, recall_score, accuracy_score
import numpy as np

"""TRAINING"""

import pandas as pd
import numpy as np
import torch
import transformers
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# --- 1. DATA LOADING AND SPLITTING ---
df = pd.read_csv('data/df_ekman_labels.csv')
emotion_cols = df.columns[1:].tolist()  # ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

# Remove rows with NaN text
df.dropna(subset=['text'], inplace=True)


# Split data into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df[emotion_cols].values,
    test_size=0.2,
    random_state=42
)

# Convert to DataFrames
X_train_df = pd.DataFrame(X_train).reset_index(drop=True)
X_test_df = pd.DataFrame(X_test).reset_index(drop=True)
y_train_df = pd.DataFrame(y_train, columns=emotion_cols)
y_test_df = pd.DataFrame(y_test, columns=emotion_cols)

# Hugging Face Datasets
train_ds = Dataset.from_pandas(pd.concat([X_train_df, y_train_df], axis=1))
test_ds  = Dataset.from_pandas(pd.concat([X_test_df, y_test_df], axis=1))

# --- 2. TOKENIZATION AND LABEL PREPARATION ---
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def clean_text(batch):
    batch["text"] = [str(x) if x is not None else "" for x in batch["text"]]
    return batch

def tokenize(batch):
    return tokenizer(
       batch["text"],
       truncation=True,
       padding="max_length",
       max_length=128
    )

def add_labels(batch):
    batch["labels"] = [
        [float(batch[col][i]) for col in emotion_cols]
        for i in range(len(batch[emotion_cols[0]]))
    ]
    return batch

# Apply transformations
train_ds = train_ds.map(clean_text, batched=True).map(tokenize, batched=True).map(add_labels, batched=True)
test_ds  = test_ds.map(clean_text, batched=True).map(tokenize, batched=True).map(add_labels, batched=True)

# Set format for PyTorch
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# --- 3. MODEL, METRICS, AND TRAINING ---
num_labels = len(emotion_cols)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    preds = (probs > 0.5).astype(int)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(labels, preds),
    }

# Training Arguments (fixed keyword)
training_args = TrainingArguments(
    output_dir="./bert-emotion",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=50,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch", # Changed from "steps" to "epoch"
    metric_for_best_model='macro_f1',
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Starting BERT Fine-Tuning...")
trainer.train()
print("BERT Fine-Tuning Complete.")