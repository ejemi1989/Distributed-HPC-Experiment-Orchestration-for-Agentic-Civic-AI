import ray
import mlflow
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score
import numpy as np

ray.init(ignore_reinit_error=True)

MODEL = "distilbert-base-uncased"

@ray.remote(num_gpus=1)
class TrainerWorker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    def train_step(self, texts, labels):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs, labels=torch.tensor(labels))
        loss = outputs.loss
        loss.backward()
        return loss.item()

    def eval_step(self, texts, labels):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        logits = self.model(**inputs).logits.detach().numpy()
        preds = np.argmax(logits, axis=1)
        return accuracy_score(labels, preds)

if __name__ == "__main__":
    mlflow.start_run()

    workers = [TrainerWorker.remote() for _ in range(4)]

    texts = [
        "City council approves housing budget.",
        "Fake news claims voting fraud."
    ]
    labels = [0, 1]

    losses = ray.get([w.train_step.remote(texts, labels) for w in workers])
    accs = ray.get([w.eval_step.remote(texts, labels) for w in workers])

    mlflow.log_metric("avg_loss", sum(losses)/len(losses))
    mlflow.log_metric("avg_accuracy", sum(accs)/len(accs))

    print("Distributed Training Complete")
