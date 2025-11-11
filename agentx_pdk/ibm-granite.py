from ray import serve
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import Request
import torch
import sys
from ray import serve

from typing import Dict, Any, List

@serve.deployment
class MedicalTextGeneratorBase:
    def __init__(self, device: int = -1):
        model_name = "ibm-granite/granite-4.0-h-350m"  # or your preferred generator model
        print(f"🔧 Loading model: {model_name} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # device selection
        self.device = (
            torch.device(f"cuda:{device}")
            if torch.cuda.is_available() and device >= 0
            else torch.device("cpu")
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Model loaded on {self.device}")

    def generate_text(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.3):
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    async def __call__(self, request) -> Dict[str, Any]:
        data = await request.json()
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", 100)
        temperature = data.get("temperature", 0.7)

        if not prompt:
            return {"error": "No prompt provided"}

        generated_text = self.generate_text(prompt, max_new_tokens, temperature)
        return {"generated_text": generated_text}


        
def model_binder(config: dict):
    device = config.get("device", -1)


    return MedicalTextGeneratorBase.bind(
        device=device,
    )