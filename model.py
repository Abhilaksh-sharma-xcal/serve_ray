from ray import serve
from transformers import pipeline
import sys
import time

@serve.deployment
class Translator:
    def __init__(self):
        self.model = pipeline("translation_en_to_fr", model="t5-small", device="cpu")
    
    def translate(self, text: str) -> str:
        model_output = self.model(text)
        return model_output[0]["translation_text"]
    
    async def __call__(self, http_request):
        text = await http_request.json()
        return self.translate(text)

# Bind the deployment
translator_app = Translator.bind()

if __name__ == "__main__":
    try:
        # Start the serve application
        serve.run(translator_app)
        
        print("Server is running. Press Ctrl+C to stop.")
        print("You can test it with: curl -X POST http://127.0.0.1:8000/ -H 'Content-Type: application/json' -d '\"Hello world\"'")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        serve.shutdown()
        sys.exit(0)