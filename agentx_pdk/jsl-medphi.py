from ray import serve
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import Request
import torch
from typing import Dict, Any, List


@serve.deployment
class MedPhi2TextGenerator:
    def __init__(self, device: int = -1):
        self.model_name = "johnsnowlabs/JSL-MedPhi2-2.7B"
        print(f"🔧 Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        self.model.eval()
        print("✅ Model loaded successfully")

    def build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Converts chat messages to model prompt
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> str:

        prompt = self.build_prompt(messages)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    async def __call__(self, request: Request) -> Dict[str, Any]:
        data = await request.json()

        messages = data.get(
            "messages",
            [{"role": "user", "content": "Hello"}]
        )

        result = self.generate(
            messages=messages,
            max_new_tokens=data.get("max_new_tokens", 256),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.95),
            top_k=data.get("top_k", 50),
        )

        return {"generated_text": result}


def model_binder(config: dict):
    device = config.get("device", -1)

    return MedPhi2TextGenerator.bind(
        device=device
    )
