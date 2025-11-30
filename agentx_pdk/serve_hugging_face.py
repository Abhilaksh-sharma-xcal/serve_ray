"""
Production-ready Ray Serve + Hugging Face universal model server.

Features:
- Correct `pipeline()` usage with task positional arg
- Device mapping (cpu / cuda / cuda:0 etc.)
- Background batching worker (aggregates requests into batches)
- Proper streaming support using `TextIteratorStreamer` when available
- Warmup using pipeline's processor/tokenizer/feature_extractor
- Support for PEFT/LoRA adapters if `peft` is installed
- HF Hub token + revision pinning
- Timeouts and cancellation
- Clean shutdown hooks and memory cleanup
- Pydantic schemas for request validation
- JSON (and base64) inputs; multipart can be added similarly

Usage:
- Deploy with `serve.run(deployable.bind(...))` or use `deploy_model()` helper at bottom.

Note: This is a high-quality template and may need small adjustments for specific models
(e.g. extremely large models, vLLM/TGI backends, custom processors).
"""

import os
import io
import json
import time
import asyncio
import base64
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field
from enum import Enum
from PIL import Image

import ray
from ray import serve
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

import torch
from transformers import pipeline, AutoTokenizer, AutoConfig
from transformers import TextIteratorStreamer

# Optional imports (PEFT), only used if available
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Configuration & Schemas
# ---------------------------

class TaskType(str, Enum):
    TEXT_GENERATION = "text-generation"
    SUMMARIZATION = "summarization"
    TEXT_CLASSIFICATION = "text-classification"
    QUESTION_ANSWERING = "question-answering"
    SENTIMENT_ANALYSIS = "sentiment-analysis"
    TRANSLATION = "translation"
    ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    IMAGE_SEGMENTATION = "image-segmentation"
    IMAGE_TO_TEXT = "image-to-text"
    VISUAL_QUESTION_ANSWERING = "visual-question-answering"
    DOCUMENT_QUESTION_ANSWERING = "document-question-answering"


class TextInput(BaseModel):
    text: Optional[str] = Field(None)
    text_pair: Optional[str] = Field(None)
    context: Optional[str] = Field(None)
    question: Optional[str] = Field(None)
    candidate_labels: Optional[List[str]] = Field(None)
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    stream: Optional[bool] = Field(False)


class ImageInput(BaseModel):
    image: str  # base64
    text: Optional[str] = Field(None)
    question: Optional[str] = Field(None)
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)


# Default params (safe sensible defaults)
TASK_DEFAULTS: Dict[TaskType, Dict[str, Any]] = {
    TaskType.TEXT_GENERATION: {"max_new_tokens": 128, "temperature": 0.7, "top_p": 0.9, "do_sample": True},
    TaskType.SUMMARIZATION: {"max_length": 130, "min_length": 30, "do_sample": False},
    TaskType.TEXT_CLASSIFICATION: {"top_k": 5},
    TaskType.QUESTION_ANSWERING: {"top_k": 1},
    TaskType.SENTIMENT_ANALYSIS: {},
    TaskType.TRANSLATION: {"max_length": 512},
    TaskType.ZERO_SHOT_CLASSIFICATION: {"multi_label": False},
    TaskType.IMAGE_CLASSIFICATION: {"top_k": 5},
}


# ---------------------------
# Utilities
# ---------------------------

def map_device_arg(device: str) -> int:
    """Map device string to HF pipeline device integer.

    Accepts: 'cpu', 'cuda', 'cuda:0', 'cuda:1', '0' (index)
    Returns: -1 for CPU, integer >= 0 for GPU
    """
    if device is None:
        return -1
    device = str(device).lower()
    if device == "cpu":
        return -1
    if device == "cuda":
        # default to first GPU
        return 0
    if device.startswith("cuda:"):
        try:
            idx = int(device.split(":", 1)[1])
            return idx
        except Exception:
            return 0
    # If it's numeric "0", "1" etc.
    try:
        return int(device)
    except Exception:
        return -1


def decode_base64_image(b64string: str) -> Image.Image:
    if "," in b64string:
        b64string = b64string.split(",", 1)[1]
    data = base64.b64decode(b64string)
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


# ---------------------------
# Batching queue items
# ---------------------------

class _QueueItem:
    def __init__(self, inputs: Any, params: Dict[str, Any], fut: asyncio.Future, is_stream: bool = False):
        self.inputs = inputs
        self.params = params
        self.future = fut
        self.is_stream = is_stream


# ---------------------------
# Model Server
# ---------------------------

@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 4, "target_ongoing_requests": 4},
    max_ongoing_requests=32,
)
class ModelServer:
    """Production-friendly universal HF server.

    Initialises pipeline properly, supports batching (background worker), streaming, warmup and cleanup.
    """

    def __init__(
        self,
        model_name: str,
        task_type: str,
        device: str = "cpu",
        batch_size: int = 4,
        batch_timeout: float = 0.05,
        timeout: float = 60.0,
        torch_dtype: Optional[str] = None,
        trust_remote_code: bool = False,
        use_fast_tokenizer: bool = True,
        model_kwargs: Optional[Dict[str, Any]] = None,
        revision: Optional[str] = None,
    ):
        self.model_name = model_name
        self.task_type = TaskType(task_type)
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.batch_timeout = float(batch_timeout)
        self.timeout = float(timeout)
        self.torch_dtype = None
        self.trust_remote_code = trust_remote_code
        self.use_fast_tokenizer = use_fast_tokenizer
        self.model_kwargs = model_kwargs or {}
        self.revision = revision

        # internal
        self._device_id = map_device_arg(self.device)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._batch_worker_task: Optional[asyncio.Task] = None
        self._closing = False

        logger.info(f"[ModelServer] Initializing {self.model_name} ({self.task_type}) on {self.device} -> device_id={self._device_id}")

        # dtype map
        if self.torch_dtype is None and torch.cuda.is_available() and self._device_id >= 0:
            # default to float16 on GPU if user didn't choose
            self.torch_dtype = torch.float16

        # Build pipeline kwargs correctly (task is positional arg but accepts as keyword too in many versions)
        pipeline_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "device": self._device_id,
            "trust_remote_code": self.trust_remote_code,
            "batch_size": self.batch_size,
        }

        if self.torch_dtype is not None:
            pipeline_kwargs["torch_dtype"] = self.torch_dtype

        # attach token and revision
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if hf_token:
            pipeline_kwargs["use_auth_token"] = hf_token
            logger.info("[ModelServer] Using HF auth token from environment")
        if self.revision:
            pipeline_kwargs["revision"] = self.revision

        # merge any user-provided model kwargs
        pipeline_kwargs.update(self.model_kwargs)

        # instantiate pipeline in a blocking thread
        try:
            # pipeline's correct calling convention: pipeline(task, **kwargs)
            self.pipe = pipeline(self.task_type.value, **pipeline_kwargs)
        except Exception as e:
            logger.exception("Failed to create HF pipeline")
            raise

        # Apply PEFT / LoRA if requested and available
        adapter_name = self.model_kwargs.get("peft_adapter") or os.environ.get("PEFT_ADAPTER")
        if adapter_name:
            if not PEFT_AVAILABLE:
                logger.warning("PEFT adapter requested but peft package not available; ignoring adapter")
            else:
                try:
                    logger.info(f"Applying PEFT adapter: {adapter_name}")
                    base_model = getattr(self.pipe, "model", None)
                    if base_model is not None:
                        peft_model = PeftModel.from_pretrained(base_model, adapter_name)
                        # replace model in pipeline
                        self.pipe.model = peft_model
                    else:
                        logger.warning("Pipeline has no .model attribute; cannot apply PEFT adapter")
                except Exception:
                    logger.exception("Failed to apply PEFT adapter")

        # get defaults for task
        self.default_params = TASK_DEFAULTS.get(self.task_type, {})

        # ensure tokenizer/processor loaded eagerly for warm first request
        try:
            _ = getattr(self.pipe, "tokenizer", None)
            _ = getattr(self.pipe, "processor", None)
            _ = getattr(self.pipe, "feature_extractor", None)
        except Exception:
            logger.debug("Tokenizer/processor access raised during eager load; continuing")

        # Perform warmup
        try:
            self._warmup()
        except Exception:
            logger.exception("Warmup failed (non-blocking)")

        # start batching worker
        loop = asyncio.get_event_loop()
        self._batch_worker_task = loop.create_task(self._batch_worker())
        logger.info("ModelServer initialized and batch worker started")

    def _warmup(self):
        logger.info("Warmup: running a small request to ensure model compiles/allocs")
        try:
            if self.task_type in (TaskType.IMAGE_CLASSIFICATION, TaskType.IMAGE_TO_TEXT,
                                  TaskType.VISUAL_QUESTION_ANSWERING, TaskType.DOCUMENT_QUESTION_ANSWERING):
                # prepare dummy image using pipeline's processor if present
                dummy_img = Image.new("RGB", (512, 512), color=(255, 255, 255))
                try:
                    # some pipelines want 'image' name, others positional
                    self.pipe(dummy_img)
                except Exception:
                    # try named
                    self.pipe(image=dummy_img)
            elif self.task_type == TaskType.QUESTION_ANSWERING:
                self.pipe({"question": "Who?", "context": "Alice and Bob"})
            else:
                # text task
                self.pipe("Hello warmup", **{"max_new_tokens": 5} if self.task_type == TaskType.TEXT_GENERATION else {})
            logger.info("Warmup finished")
        except Exception:
            logger.exception("Warmup encountered an error; continuing")

    async def _batch_worker(self):
        """Background worker that accumulates requests into batches and runs pipeline once."""
        logger.info("Batch worker running")
        while not self._closing:
            try:
                # wait for first item
                item: _QueueItem = await self._queue.get()
                items = [item]
                start = time.time()

                # accumulate until batch_size or timeout
                while len(items) < self.batch_size and (time.time() - start) < self.batch_timeout:
                    try:
                        item2 = self._queue.get_nowait()
                        items.append(item2)
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0)  # yield

                # prepare batch inputs and params
                batch_inputs = [it.inputs for it in items]
                batch_params = [it.params for it in items]

                # run inference in thread to avoid blocking event loop
                try:
                    results = await asyncio.to_thread(self._run_pipeline_batch_sync, batch_inputs, batch_params)
                except Exception as e:
                    logger.exception("Batch inference exception")
                    # set exception on all futures
                    for it in items:
                        if not it.future.done():
                            it.future.set_exception(e)
                    continue

                # deliver results to futures
                for it, res in zip(items, results):
                    if not it.future.done():
                        it.future.set_result(res)

            except Exception as e:
                logger.exception("Batch worker top-level error")
                await asyncio.sleep(0.01)

        logger.info("Batch worker exiting")

    def _run_pipeline_batch_sync(self, inputs: List[Any], params_list: List[Dict[str, Any]]) -> List[Any]:
        """Synchronous batch runner: tries to use pipeline with batched inputs where possible.

        The strategy:
        - If pipeline accepts list input (many HF pipelines do), pass list
        - otherwise run single-item calls and collect
        """
        results: List[Any] = []

        # try list input
        try:
            single_params = self.default_params.copy()
            single_params.update(params_list[0] if params_list else {})
            # If all params are identical, we can pass them as one dict
            identical = all(p == params_list[0] for p in params_list)

            if identical:
                # pass list of inputs + single params
                res = self.pipe(inputs, **single_params)
                # The pipeline may return a list-of-results or a single object; normalize
                if isinstance(res, list) and len(res) == len(inputs):
                    return res
                # otherwise interpret as single result repeated
                return [res for _ in inputs]
        except Exception:
            # fall back to itemwise
            logger.debug("Batch list-call not supported, falling back to itemwise")

        # itemwise fallback
        for inp, params in zip(inputs, params_list):
            try:
                merged = self.default_params.copy()
                merged.update(params or {})
                if isinstance(inp, dict):
                    res = self.pipe(**inp, **merged)
                else:
                    res = self.pipe(inp, **merged)
                results.append(res)
            except Exception as e:
                logger.exception("Error running item in batch")
                results.append({"error": str(e)})

        return results

    async def __call__(self, request: Request) -> Union[JSONResponse, StreamingResponse]:
        """Entrypoint for Ray Serve - accepts JSON POST requests only (for now)."""
        # Only accept POST
        if request.method != "POST":
            return JSONResponse(status_code=405, content={"success": False, "error": "Method not allowed"})

        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type:
            return JSONResponse(status_code=415, content={"success": False, "error": "Only application/json supported in this deployment"})

        try:
            data = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid JSON"})

        # Prepare input and params depending on task
        try:
            is_image_task = self.task_type in (
                TaskType.IMAGE_CLASSIFICATION,
                TaskType.OBJECT_DETECTION,
                TaskType.IMAGE_SEGMENTATION,
                TaskType.IMAGE_TO_TEXT,
                TaskType.VISUAL_QUESTION_ANSWERING,
                TaskType.DOCUMENT_QUESTION_ANSWERING,
            )

            if is_image_task:
                inp = ImageInput(**data)
                image = decode_base64_image(inp.image)
                if self.task_type in (TaskType.VISUAL_QUESTION_ANSWERING, TaskType.DOCUMENT_QUESTION_ANSWERING):
                    pipeline_input = {"image": image, "question": inp.question}
                elif self.task_type == TaskType.IMAGE_TO_TEXT and inp.text:
                    pipeline_input = {"image": image, "prompt": inp.text}
                else:
                    pipeline_input = image
                params = (TASK_DEFAULTS.get(self.task_type, {}).copy() if not inp.params else TASK_DEFAULTS.get(self.task_type, {}).copy())
                params.update(inp.params or {})
                stream = False
            else:
                txt = TextInput(**data)
                params = TASK_DEFAULTS.get(self.task_type, {}).copy()
                params.update(txt.params or {})
                stream = bool(txt.stream and self.task_type == TaskType.TEXT_GENERATION)

                # craft pipeline input depending on task
                if self.task_type == TaskType.QUESTION_ANSWERING:
                    if not txt.question or not txt.context:
                        raise ValueError("question and context required for question-answering")
                    pipeline_input = {"question": txt.question, "context": txt.context}
                elif self.task_type == TaskType.ZERO_SHOT_CLASSIFICATION:
                    if not txt.candidate_labels:
                        raise ValueError("candidate_labels required for zero-shot classification")
                    pipeline_input = {"sequences": txt.text, "candidate_labels": txt.candidate_labels}
                elif self.task_type == TaskType.TEXT_CLASSIFICATION:
                    pipeline_input = txt.text if txt.text is not None else ""
                else:
                    pipeline_input = txt.text if txt.text is not None else ""

            # handle streaming
            if stream:
                # create streamer and run generate in background
                return await self._start_streaming(pipeline_input, params)

            # prepare queue future and enqueue
            loop = asyncio.get_event_loop()
            fut: asyncio.Future = loop.create_future()
            q_item = _QueueItem(pipeline_input, params, fut)
            await self._queue.put(q_item)

            # wait for result with timeout
            try:
                res = await asyncio.wait_for(fut, timeout=self.timeout)
                return JSONResponse(status_code=200, content={"success": True, "result": res, "model": self.model_name, "task": self.task_type.value})
            except asyncio.TimeoutError:
                if not fut.done():
                    fut.cancel()
                logger.error("Inference timed out")
                return JSONResponse(status_code=504, content={"success": False, "error": "Inference timed out"})

        except ValueError as e:
            logger.error("Validation error: %s", str(e))
            return JSONResponse(status_code=400, content={"success": False, "error": str(e), "error_type": "validation"})
        except Exception as e:
            logger.exception("Unhandled error in request handling")
            return JSONResponse(status_code=500, content={"success": False, "error": str(e), "error_type": "inference"})

    async def _start_streaming(self, inputs: Any, params: Dict[str, Any]) -> StreamingResponse:
        """Start SSE streaming for text generation using TextIteratorStreamer when available."""
        # Only support simple text input for streaming
        if self.task_type != TaskType.TEXT_GENERATION:
            raise ValueError("Streaming supported only for text-generation")

        if not hasattr(self.pipe, "model") or not hasattr(self.pipe, "tokenizer"):
            raise ValueError("Underlying pipeline doesn't expose model/tokenizer needed for streaming")

        tokenizer = self.pipe.tokenizer
        model = self.pipe.model

        # build streamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=False, decode_kwargs={"skip_special_tokens": True})

        def generate_in_thread():
            try:
                # prepare generate kwargs
                gen_kwargs = {**self.default_params}
                gen_kwargs.update(params or {})
                # call generate with streamer (this blocks current thread)
                # For encoder-decoder models, pass input_ids differently
                if isinstance(inputs, str):
                    # tokenize to input ids on model device
                    batch = tokenizer(inputs, return_tensors="pt")
                    input_ids = batch.get("input_ids").to(model.device)
                    attention_mask = batch.get("attention_mask").to(model.device) if batch.get("attention_mask") is not None else None
                    model.generate(input_ids=input_ids, attention_mask=attention_mask, streamer=streamer, **{k: v for k, v in gen_kwargs.items()})
                else:
                    # fallback: let pipeline handle streaming if it supports streamer arg (rare)
                    self.pipe(inputs, streamer=streamer, **gen_kwargs)
            except Exception:
                logger.exception("Streaming generation failed")
                # push error sentinel
                streamer.on_text("[STREAM_ERROR]")

        loop = asyncio.get_event_loop()
        # run generate in a background thread (blocking call)
        await loop.run_in_executor(None, generate_in_thread)

        async def event_generator():
            try:
                for token in streamer:
                    # streamer yields decoded strings
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception:
                yield f"data: {json.dumps({'error': 'stream_error'})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    async def health(self) -> Dict[str, Any]:
        return {"status": "healthy", "model": self.model_name, "task": self.task_type.value, "device": self.device}

    async def __del__(self):
        # cleanup
        self._closing = True
        if self._batch_worker_task:
            try:
                self._batch_worker_task.cancel()
            except Exception:
                pass
        # free memory
        try:
            if hasattr(self, "pipe") and hasattr(self.pipe, "model"):
                del self.pipe.model
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------
# Deployment helper
# ---------------------------

def deploy_model(
    model_name: str,
    task_type: str,
    deployment_name: str,
    device: str = "cpu",
    num_gpus: float = 0,
    num_replicas: int = 1,
    **kwargs,
) -> serve.deployment:
    """Create a deployment binding for Ray Serve."""
    return ModelServer.options(
        name=deployment_name,
        ray_actor_options={"num_gpus": num_gpus},
        num_replicas=num_replicas,
    ).bind(model_name=model_name, task_type=task_type, device=device, **kwargs)


# ---------------------------
# Example: local run
# ---------------------------
# if __name__ == "__main__":
#     ray.init(ignore_reinit_error=True)

#     # Example: simple gpt2 text generation
#     text_dep = deploy_model(
#         model_name="gpt2",
#         task_type="text-generation",
#         deployment_name="gpt2-prod",
#         device="cpu",
#         batch_size=4,
#         batch_timeout=0.05,
#     )

#     serve.run(text_dep, name="text_gen", route_prefix="/generate")
#     print("Deployment started at /generate")
