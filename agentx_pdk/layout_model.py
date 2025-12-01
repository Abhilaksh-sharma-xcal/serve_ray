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
import base64


DEFAULT_PARAMS = {
    "text-generation": {"max_new_tokens": 50, "temperature": 0.7, "do_sample": True},
    "summarization": {"max_length": 128, "min_length": 32},
    "text-classification": {},
    "question-answering": {},
    "sentiment-analysis": {},
    "object-detection": {},
}


@serve.deployment
class ModelServer:
    def __init__(self, model_name: str, task_type: str, device: int = -1, **kwargs):
        self.task_type = task_type
        self.device = device
        pipe_kwargs = {
            "task": task_type,
            "model": model_name,
            "device": device,
            **kwargs
        }
        
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token:
            print("[ModelServer] Using Hugging Face token from environment")
            pipe_kwargs["use_auth_token"] = token
        print(f"[ModelServer] Loading '{model_name}' for task '{task_type}' on device {device}")
        self.pipe = pipeline(**pipe_kwargs)
        
        self.default_params = DEFAULT_PARAMS.get(task_type, {})

    def _decode_image(self, image_input: str) -> Image.Image:
        """
        Decode image from various formats:
        - Base64 string
        - Base64 with data URI prefix (data:image/jpeg;base64,...)
        - URL (handled by pipeline directly)
        """
        if image_input.startswith(('http://', 'https://')):
            # Let the pipeline handle URLs directly
            return image_input
        
        # Remove data URI prefix if present
        if image_input.startswith('data:image'):
            # Format: data:image/jpeg;base64,<base64_string>
            image_input = image_input.split(',', 1)[1]
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_input)
            return Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Failed to decode image: {str(e)}")

    async def __call__(self, request) -> Dict[str, Any]:
        try:
            content_type = request.headers.get("content-type", "")
            params = self.default_params.copy()
            pipeline_inputs = {}

            if "application/json" in content_type:
                data = await request.json()
                
                # Handle text inputs
                if "text" in data:
                    pipeline_inputs["text"] = data["text"]
                
                # Handle image inputs (NEW!)
                if "image" in data:
                    pipeline_inputs["image"] = self._decode_image(data["image"])
                
                # Handle generic "inputs" field (common in HF API)
                if "inputs" in data and not pipeline_inputs:
                    inputs = data["inputs"]
                    
                    # Check if it's an image (base64 or URL)
                    if isinstance(inputs, str):
                        if inputs.startswith(('http://', 'https://', 'data:image')) or \
                           (len(inputs) > 100 and self._looks_like_base64(inputs)):
                            # It's likely an image
                            pipeline_inputs["image"] = self._decode_image(inputs)
                        else:
                            # It's text
                            pipeline_inputs["text"] = inputs
                    elif isinstance(inputs, dict):
                        # Handle nested inputs
                        if "image" in inputs:
                            pipeline_inputs["image"] = self._decode_image(inputs["image"])
                        elif "text" in inputs:
                            pipeline_inputs["text"] = inputs["text"]
                
                # Handle question-answering specific inputs
                if "question" in data:
                    pipeline_inputs["question"] = data["question"]
                if "context" in data:
                    pipeline_inputs["context"] = data["context"]
                
                # Update parameters
                params.update(data.get("params", {}))
                params.update(data.get("parameters", {}))
                
                # Execute pipeline
                if not pipeline_inputs:
                    raise ValueError("No valid input fields found in JSON request.")
                
                # Try dict input first, fallback to single input
                try:
                    result = self.pipe(**pipeline_inputs, **params)
                except Exception:
                    first_value = next(iter(pipeline_inputs.values()))
                    result = self.pipe(first_value, **params)

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

    def _looks_like_base64(self, s: str) -> bool:
        """Check if string looks like base64"""
        if len(s) < 100:  # Too short to be an image
            return False
        # Base64 only contains these characters
        import string
        allowed = set(string.ascii_letters + string.digits + '+/=')
        return all(c in allowed for c in s[:100])


def model_binder(config: dict):
    device = config.get("device", -1)
    model_name = config.get("model_name")
    task_type = config.get("task_type")

    extra_kwargs = {
        k: v
        for k, v in config.items()
        if k not in {"model_name", "task_type", "device"}
    }

    return ModelServer.bind(
        model_name=model_name,
        task_type=task_type,
        device=device,
        **extra_kwargs
    )