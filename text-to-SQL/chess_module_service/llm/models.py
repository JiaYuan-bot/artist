import os

from .engine_configs import ENGINE_CONFIGS

from google import genai
from google.genai import types

# GCP_PROJECT = os.getenv("GCP_PROJECT")
# GCP_REGION = os.getenv("GCP_REGION")
GCP_PROJECT = 'dbgroup'
GCP_REGION = 'us-central1'


def gemini_api_call_with_config(model_name: str, prompt: str):
    if model_name not in ENGINE_CONFIGS:
        raise ValueError(f"Model {model_name} not found in ENGINE_CONFIGS")

    params = ENGINE_CONFIGS[model_name]["params"]

    client = genai.Client(
        vertexai=True,
        project=GCP_PROJECT,
        location=GCP_REGION,
    )

    # Build generation config from your params
    generation_config = {
        "temperature": params.get("temperature", 0),
        "top_p": params.get("top_p", 0.5),
        "top_k": params.get("top_k", 3),
    }

    response = client.models.generate_content(model=params["model"],
                                              contents=[prompt],
                                              config=generation_config)

    # print(response.text)
    return response.text


# Usage
# response = gemini_api_call_with_config("gemini-2.5-flash", "Your prompt here")
