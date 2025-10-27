import os
import traceback
import ray
from ray import serve
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from transformers import AutoTokenizer 

# Initialize FastAPI app for ingress
app = FastAPI()

@serve.deployment(
    ray_actor_options={"num_cpus": 4, "num_gpus": 1},
    max_ongoing_requests=10,
)
@serve.ingress(app)
class VLLMModelServer:
    def __init__(self, model_name: str, **kwargs):
        """Initialize vLLM Engine and OpenAI-compatible serving layer."""
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            print("[VLLM] Using Hugging Face token from env")


        engine_args = AsyncEngineArgs(model=model_name, **kwargs)

        # Initialize async vLLM engine
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        chat_template = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            chat_template = getattr(tokenizer, "chat_template", None)
            if not chat_template:
                raise ValueError("No chat template found; using fallback.")
            print(f"[VLLM] Loaded chat template from tokenizer for {model_name}.")
        except Exception as e:
            print(f"[VLLM][Warning] {e}")
            chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}System: {{ message['content'] }}\n"
                "{% elif message['role'] == 'user' %}User: {{ message['content'] }}\n"
                "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n"
                "{% endif %}"
                "{% endfor %}\nAssistant:"
            )
            print(f"[VLLM] Using fallback chat template for {model_name}.")

        
        # Configure OpenAI-compatible serving layer
        base_model_paths = [BaseModelPath(name=model_name, model_path=model_name)]
        openai_serving_models = OpenAIServingModels(
            engine_client=self.engine,
            model_config=self.engine.model_config,
            base_model_paths=base_model_paths,
            lora_modules=None,
        )

        self.openai_serving_chat = OpenAIServingChat(
            engine_client=self.engine,
            model_config=self.engine.model_config,
            models=openai_serving_models,
            response_role="assistant",
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="string",
        )

        print("[VLLM] OpenAI-compatible serving initialized successfully.")

    # === /v1/chat/completions ===
    @app.post("/v1/chat/completions")
    async def chat_completions(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-style chat completion endpoint."""
        try:
            generator = await self.openai_serving_chat.create_chat_completion(
                request, raw_request
            )

            if request.stream:
                return StreamingResponse(
                    content=generator, media_type="text/event-stream"
                )
            else:
                return JSONResponse(content=generator.model_dump())
        except Exception as e:
            print(f"[VLLM][Error] {traceback.format_exc()}")
            return JSONResponse(content={"error": str(e)}, status_code=500)

    # === /v1/models ===
    @app.get("/v1/models")
    async def models(self):
        """List available models."""
        models = await self.openai_serving_chat.show_available_models()
        return JSONResponse(content=models.model_dump())

    # === /health ===
    @app.get("/health")
    async def health(self):
        """Simple health check."""
        return {"status": "healthy"}


def model_binder(config: dict):
    model_name = config.get("model_name")
    tensor_parallel_size = config.get("tensor_parallel_size", 1)

    extra_kwargs = {
        k: v
        for k, v in config.items()
        if k not in ["model_name", "tensor_parallel_size"]
    }

    return VLLMModelServer.bind(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        **extra_kwargs,
    )