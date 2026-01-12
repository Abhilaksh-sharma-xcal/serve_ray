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

        # Force use microsoft/phi-2 config as fallback
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                resume_download=True,
            )
        except Exception as e:
            print(f"⚠️  Failed to load tokenizer: {e}")
            print("🔄 Falling back to microsoft/phi-2 tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/phi-2",
                trust_remote_code=True
            )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                resume_download=True,
            )
        except Exception as e:
            print(f"⚠️  Failed to load model with auto config: {e}")
            print("🔄 Trying with explicit config from microsoft/phi-2")
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                "microsoft/phi-2",
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                resume_download=True,
            )

        self.model.eval()
        print("✅ Model loaded successfully")

    def build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Converts chat messages to model prompt
        """
        # Check if tokenizer has chat template
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"⚠️ Chat template failed: {e}, using simple format")

        # Fallback: Build prompt manually for Phi-2 format
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"Instruct: {content}\n"
            elif role == "assistant":
                prompt += f"Output: {content}\n"
        prompt += "Output:"
        return prompt

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

        # Support both "prompt" (simple string) and "messages" (chat format)
        if "prompt" in data:
            messages = [{"role": "user", "content": data["prompt"]}]
        else:
            messages = data.get(
                "messages",
                [{"role": "user", "content": "Hello"}]
            )

        # Handle max_tokens or max_new_tokens
        max_new_tokens = data.get("max_new_tokens") or data.get("max_tokens", 256)

        result = self.generate(
            messages=messages,
            max_new_tokens=max_new_tokens,
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
