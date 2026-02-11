import pandas as pd
import nltk
from datasets import Dataset
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from sklearn.metrics import confusion_matrix
import os


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

df=pd.read_csv("C:\\Users\\Shin Yee\\Downloads\\archive (11)\\Training_Essay_Data.csv")

#preprocessing
#transform lower case
df["text"]=df["text"].str.lower()
#remove HTML, noise
df["text"]=df["text"].str.replace(r"<.*>"," ",regex=True)
df["text"] = df["text"].str.replace(r"[^a-zA-Z0-9\s]", " ", regex=True)
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

#Tokenization
df["tokens"] = df["text"].apply(word_tokenize)
#lemmatization
lemmatizer = WordNetLemmatizer()

df = df.dropna(subset=["text"])

#turn to text data type
df["text"] = df["text"].astype(str)

df.to_csv(r"C:\Users\Shin Yee\Downloads\archive (11)\Training_Essay_Data_clean.csv", index=False)


#model training
df = pd.read_csv(r"C:\Users\Shin Yee\Downloads\archive (11)\Training_Essay_Data_clean.csv")
def clean_text_column(df, col="text"):
    # Normalize: convert to string / flatten list into text
    df[col] = df[col].apply(
        lambda x: " ".join(x) if isinstance(x, (list, tuple))
        else str(x)
    )
    # Remove empty strings
    df = df[df[col].str.strip() != ""]
    return df

df = clean_text_column(df, col="text")

# Rename target column to 'label' (required by Hugging Face Trainer)
df.rename(columns={"generated": "label"}, inplace=True)

# transform HuggingFace Dataset
dataset = Dataset.from_pandas(df)
#split test, train
dataset = dataset.train_test_split(test_size=0.2, seed=42)
# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess(batch):
    texts = [str(t) for t in batch["text"]]
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

encoded_dataset = dataset.map(preprocess, batched=True)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2  )

# model evalution
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
    f1_score = f1.compute(predictions=preds, references=labels)["f1"]
    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1_score,
        "confusion_matrix": cm.tolist()
    }

# argument
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# model train
trainer.train()

metrics = trainer.evaluate(encoded_dataset["test"])
print("Final Test Accuracy:", metrics["eval_accuracy"])
print("Final Test F1:", metrics["eval_f1"])
print("Confusion Matrix:", metrics["eval_confusion_matrix"])