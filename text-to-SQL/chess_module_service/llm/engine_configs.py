from langchain_google_vertexai import VertexAI
from google.cloud import aiplatform
from typing import Dict, Any
import vertexai
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
import os

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
    HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
    HarmBlockThreshold.BLOCK_NONE,
}

GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_REGION = os.getenv("GCP_REGION")
# GCP_CREDENTIALS = os.getenv("GCP_CREDENTIALS")

if GCP_PROJECT and GCP_REGION:
    aiplatform.init(
        project=GCP_PROJECT,
        location=GCP_REGION,
    )
    vertexai.init(
        project=GCP_PROJECT,
        location=GCP_REGION,
    )
"""
This module defines configurations for various language models using the langchain library.
Each configuration includes a constructor, parameters, and an optional preprocessing function.
"""

ENGINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gemini-2.5-flash-lite": {
        "constructor": VertexAI,
        "params": {
            "model": "gemini-2.5-flash-lite",
            # "temperature": 0,
            # "top_p": 0.5,
            # "top_k": 3,
            # "seed": 1.0,
            # "safety_settings": safety_settings
        }
    },
    "gemini-2.5-flash": {
        "constructor": VertexAI,
        "params": {
            "model": "gemini-2.5-flash",
            # "temperature": 0,
            # "top_p": 0.5,
            # "top_k": 3,
            # "seed": 1.0,
            # "safety_settings": safety_settings
        }
    },
    "gemini-2.0-flash-lite": {
        "constructor": VertexAI,
        "params": {
            "model": "gemini-2.0-flash-lite-001",
            # "temperature": 0,
            # "top_p": 0.5,
            # "top_k": 3,
            # "seed": 1.0,
            # "safety_settings": safety_settings
        }
    },
    "gemini-2.0-flash": {
        "constructor": VertexAI,
        "params": {
            "model": "gemini-2.0-flash-001",
            # "temperature": 0,
            # "top_p": 0.5,
            # "top_k": 3,
            # "seed": 1.0,
            # "safety_settings": safety_settings
        }
    },
    "gemini-2.5-pro": {
        "constructor": VertexAI,
        "params": {
            "model": "gemini-2.5-pro",
            # "temperature": 0,
            # "top_p": 0.5,
            # "top_k": 3,
            # "seed": 1.0,
            # "safety_settings": safety_settings
        }
    }
}
