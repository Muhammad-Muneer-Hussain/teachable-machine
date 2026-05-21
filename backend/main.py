import os
import uuid
import io
import pickle
from typing import List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.linear_model import LogisticRegression
from PIL import Image

app = FastAPI(title="Teachable Machine Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET_DIR = "dataset"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = MobileNet_V3_Small_Weights.DEFAULT
feature_extractor = mobilenet_v3_small(weights=weights)

import torch.nn as nn
feature_extractor.classifier = nn.Identity()
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

# Exact Preprocessing Pipeline (224x224, Normalize)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_bytes: bytes) -> torch.Tensor:
    """Helper function to load image, preprocess, and extract features."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = feature_extractor(tensor)
        return features.squeeze().cpu()
    except Exception as e:
        raise ValueError(f"Invalid image format: {str(e)}")
