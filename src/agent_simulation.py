import ray
import random
from transformers import pipeline

ray.init(ignore_reinit_error=True)

@ray.remote
class CivicAgent:
    def __init__(self, name):
        self.name = name
        self.pipe = pipeline("text-classification")

    def analyze(self, text):
        result = self.pipe(text)[0]
        return {"agent": self.name, "analysis": result}

if __name__ == "__main__":
    agents = [CivicAgent.remote(f"agent-{i}") for i in range(8)]

    tasks = [
        "False report claims hospital shutdown",
        "Public consultation on water infrastructure"
    ]

    futures = [random.choice(agents).analyze.remote(t) for t in tasks]
    results = ray.get(futures)

    for r in results:
        print(r)
