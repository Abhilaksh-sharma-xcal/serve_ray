import requests

url = "http://localhost:8000/fb"  # default Serve endpoint
# data = {
#     "prompt": "Hello, world!",
#     "params": {  # must be a dict
#         "temperature": 0.7,
#         "top_p": 0.9,
#         "max_tokens": 10
#     }
# }
data = {}

resp = requests.post(url, json=data)

# Print status and response safely
print("Status code:", resp.status_code)
print("Response text:", resp.text)

try:
    print("JSON response:", resp.json())
except Exception as e:
    print("Failed to parse JSON:", e)
