import ray
from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

@serve.deployment()
class VLLMDeployment:
    def __init__(self, model_name: str):
        # Configure and initialize the vLLM engine
        engine_args = AsyncEngineArgs(model=model_name, tensor_parallel_size=1)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.request_counter = 0

    async def generate(self, prompt: str):
        # Define generation parameters
        sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_tokens": 256}
        
        # Stream results from the vLLM engine
        results_generator = self.engine.generate(prompt, sampling_params, request_id=f"req-{self.request_counter}")
        self.request_counter += 1
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        return final_output.outputs[0].text

    # This makes the deployment callable via HTTP
    async def __call__(self, request) -> str:
        data = await request.json()
        prompt = data.get("prompt", "What is the meaning of life?")
        return await self.generate(prompt)


def model_binder(config: dict):

    model_name = config.get("model_name", -1)

    return VLLMDeployment.bind(model_name=model_name)

# Define and run the deployment
# deployment = VLLMDeployment.bind(model_name="meta-llama/Llama-2-7b-chat-hf")
# serve.run(deployment)

# You can then send a request using curl or another client:
# curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Tell me a story about a brave knight."}' http://127.0.0.1:8000/