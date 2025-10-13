from ray import serve

@serve.deployment
class NaiveModel:
    """A simple model that checks for keywords."""
    def analyze(self, text: str) -> str:
        if "happy" in text.lower() or "love" in text.lower():
            return "POSITIVE"
        elif "sad" in text.lower() or "hate" in text.lower():
            return "NEGATIVE"
        else:
            return "NEUTRAL"