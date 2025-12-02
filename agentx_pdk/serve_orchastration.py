from transformers import pipeline
import ray
from ray import serve
from fastapi import FastAPI
from typing import Dict, Any, List


# -------------------------------------------------------
# ChatPipeline: Wraps HF transformers pipeline
# -------------------------------------------------------
class ChatPipeline:
    def __init__(self, model_name="nvidia/Orchestrator-8B"):
        # HF pipeline for text generation
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.2,
        )

    def _format_messages(self, messages):
        """Convert chat messages to a plain text prompt."""
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        return prompt

    def __call__(self, messages):
        """Allows: chat(messages)"""
        prompt = self._format_messages(messages)
        output = self.pipe(prompt)[0]["generated_text"]
        response = output[len(prompt):].strip()
        return {"role": "assistant", "content": response}


# -------------------------------------------------------
# Ray Serve Deployment Wrapper
# -------------------------------------------------------
@serve.deployment()
class ModelServer:
    def __init__(self, model_name, task_type, device=-1, **kwargs):
        self.model_name = model_name
        self.task_type = task_type
        self.device = device
        self.kwargs = kwargs

        # Load the chat pipeline
        self.model = ChatPipeline(model_name=model_name)

    async def __call__(self, request):
        """
        FastAPI Integration:
        Request JSON:
        {
            "messages": [ {"role": "user", "content": "..."} ]
        }
        """
        data = await request.json()
        messages = data.get("messages", [])
        result = self.model(messages)
        return result


# -------------------------------------------------------
# Configurable Binder (Factory for Ray Serve)
# -------------------------------------------------------
def model_binder(config: dict):
    device = config.get("device", -1)
    model_name = config.get("model_name")
    task_type = config.get("task_type")

    extra_kwargs = {
        k: v
        for k, v in config.items()
        if k not in {"model_name", "task_type", "device"}
    }

    # Returns a Ray Serve DAG node
    return ModelServer.bind(
        model_name=model_name,
        task_type=task_type,
        device=device,
        **extra_kwargs
    )
