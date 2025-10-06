import ray
from transformers import pipeline
from typing import Dict, Any, List
from ray import serve
import time

DEFAULT_PARAMS = {
    "text-generation": {"max_new_tokens": 50, "temperature": 0.7, "do_sample": True},
    "summarization": {"max_length": 128, "min_length": 32},
    "text-classification": {},
    "question-answering": {},
    "sentiment-analysis": {},
}


@serve.deployment
class ModelServer:
    def __init__(self, model_name: str, task_type: str, device: int = -1):
        self.pipe = pipeline(task=task_type, model=model_name, device=device)
        self.task_type = task_type
        self.default_params = DEFAULT_PARAMS.get(task_type, {})

    async def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data = await request.json()
            text = data["text"]
            params = data.get("params", {})
            full_params = {**self.default_params, **params}
            result = self.pipe(text, **full_params)
            return {"success": True, "result": result, "task": self.task_type}
        except Exception as e:
            return {"success": False, "error": f"{str(e)}", "task": self.task_type}


def model_binder(config: dict):
    device = config.get("device", -1)
    return ModelServer.bind(device=device)

# import asyncio

# if __name__ == "__main__":
#     # Staring the server on 0.0.0.0 host and 8000 port "
#     serve.start(http_options={"host": "0.0.0.0", "port": 8000})

#     # Can be accessed throgh "localhost/gpt2"
#     serve.run(
#         ModelServer.options(name="gpt2").bind(
#             model_name="gpt2",
#             task_type="text-generation"
#         ),
#         name="gpt2-service",
#         route_prefix="/gpt2",
#         )

#     # Can be accessed throgh "localhost/qa"
#     serve.run(
#         ModelServer.options(name="qa-model").bind(
#             model_name="distilbert-base-cased-distilled-squad",
#             task_type="question-answering"
#         ),
#         name="qa-service",
#         route_prefix="/qa",
#         )

#     # Can be accessed throgh "localhost/setiment
#     serve.run(
#         ModelServer.bind(
#             model_name="distilbert-base-uncased-finetuned-sst-2-english",
#             task_type="sentiment-analysis"
#             ),
#         name="Sentiment analysis",
#         route_prefix="/sentiment"
#         )

#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         print("Shutting down...")

#     # asyncio.get_event_loop().run_forever()
#     # asyncio.run(asyncio.sleep(float("inf")))
