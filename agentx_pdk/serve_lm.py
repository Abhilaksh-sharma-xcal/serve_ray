import ray
from transformers import pipeline
from typing import Dict, Any, List
from ray import serve
import time
import os
from fastapi import UploadFile
import io
import json
import torch
from PIL import Image


DEFAULT_PARAMS = {
    "text-generation": {"max_new_tokens": 50, "temperature": 0.7, "do_sample": True},
    "summarization": {"max_length": 128, "min_length": 32},
    "text-classification": {},
    "question-answering": {},
    "sentiment-analysis": {},
}


@serve.deployment
class ModelServer:
    def __init__(self, model_name: str, task_type: str, device: int = -1, **kwargs):
        self.task_type = task_type
        self.device = device
        pipe_kwargs = {
            "task": task_type,
            "model":model_name,
            "device":device,
            **kwargs
        }
        
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token:
            print("[ModelServer] Using Hugging Face token from environment")
            pipe_kwargs["use_auth_token"] = token
        print(f"[ModelServer] Loading '{model_name}' for task '{task_type}' on device {device}")
        self.pipe = pipeline(**pipe_kwargs)
        
        self.default_params = DEFAULT_PARAMS.get(task_type, {})

    async def __call__(self, request) -> Dict[str, Any]:
        try:
            content_type = request.headers.get("content-type", "")
            params = self.default_params.copy()
            pipeline_inputs = {}

            if "application/json" in content_type:
                data = await request.json()
                text = data.get("text", "")
                params.update(data.get("params", {}))
                result = self.pipe(text, **params)

            elif "multipart/form-data" in content_type:
                form = await request.form()

                for key, value in form.items():
                    if isinstance(value, UploadFile):
                        file_bytes = await value.read()
                        if key == "image":
                            pipeline_inputs["image"] = Image.open(io.BytesIO(file_bytes))
                        else:
                            pipeline_inputs[key] = file_bytes
                    else:
                        if key == "params":
                            params.update(json.loads(value))
                        else:
                            pipeline_inputs[key] = value

                if not pipeline_inputs:
                    raise ValueError("No valid input fields found.")

                # Try dict input first, fallback to single input
                try:
                    result = self.pipe(**pipeline_inputs, **params)
                except Exception:
                    first_value = next(iter(pipeline_inputs.values()))
                    result = self.pipe(first_value, **params)

            else:
                return {"success": False, "error": "Unsupported content-type."}

            return {"success": True, "result": result, "task": self.task_type}

        except Exception as e:
            return {"success": False, "error": str(e), "task": self.task_type}


def model_binder(config: dict):
    device = config.get("device", -1)
    model_name = config.get("model_name")
    task_type = config.get("task_type")

    extra_kwargs = {
        k: v
        for k, v in config.items()
        if k not in {"model_name", "task_type", "device"}
    }

    # Pass them into ModelServer.bind
    return ModelServer.bind(
        model_name=model_name,
        task_type=task_type,
        device=device,
        **extra_kwargs
    )

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
