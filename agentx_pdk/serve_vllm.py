import ray
from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams
from typing import Dict, Any
import os
import json
import traceback

@serve.deployment()
class VLLMModelServer:
    def __init__(self, model_name: str,device: str = "cuda", **kwargs):
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token  
            print("[VLLM] Using Hugging Face token from env")


        engine_args = AsyncEngineArgs(model=model_name,device=device, **kwargs)
        print(f"[VLLM] Initializing model '{model_name}' with args: {kwargs}")

        # Initialize the vLLM async engine
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.request_counter = 0

    async def generate(self, prompt: str, sampling_params: Dict[str, Any] = None) -> str:

        params_dict = sampling_params or {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 256,
        }
        
        sampling_params_obj = SamplingParams(**params_dict)

        try:
            request_id = f"req-{self.request_counter}"
            self.request_counter += 1

            results_generator = self.engine.generate(prompt, sampling_params_obj, request_id=request_id)
            final_output = None

            async for request_output in results_generator:
                final_output = request_output 

            if not final_output or not final_output.outputs:
                raise ValueError("No output generated.")

            return final_output.outputs[0].text

        except Exception as e:
            print(f"[VLLM][Error] {traceback.format_exc()}")
            raise RuntimeError(f"Generation failed: {e}")

    async def __call__(self, request) -> Dict[str, Any]:
        try:
            data = await request.json()
            prompt = data.get("prompt", "")
            params = data.get("params", {})

            if not prompt:
                return {"success": False, "error": "Missing 'prompt' field."}

            result = await self.generate(prompt, sampling_params=params)

            return {
                "success": True,
                "result": result,
                "model": self.engine.model_config.model,
            }

        except Exception as e:
            print(f"[VLLM][Request Error] {traceback.format_exc()}")
            return {"success": False, "error": str(e)}


def model_binder(config: dict):
    model_name = config.get("model_name")
    tensor_parallel_size=config.get("tensor_parallel_size"),
    device=config.get("device"),
    extra_kwargs = {k: v for k, v in config.items() if k != "model_name"}

    return VLLMModelServer.bind(model_name=model_name,tensor_parallel_size=tensor_parallel_size,
        device=device, **extra_kwargs)
