import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("HF_API_KEY")
headers = {"Authorization": f"Bearer {api_key}"}

model = "gianlab/swin-tiny-patch4-window7-224-finetuned-plantdisease"

urls = [
    f"https://huggingface.co/api/inference/models/{model}",
    f"https://router.huggingface.co/v1/models/{model}",
    f"https://router.huggingface.co/hf-inference/models/{model}"
]

for url in urls:
    print(f"Testing URL: {url}")
    try:
        response = requests.post(url, headers=headers, json={})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 20)
