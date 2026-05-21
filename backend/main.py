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


@app.post("/upload-sample")
async def upload_sample(class_name: str = Form(...), files: List[UploadFile] = File(...)):
    if not class_name.strip():
        raise HTTPException(status_code=400, detail="Class name cannot be empty.")
        
    class_dir = os.path.join(DATASET_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    saved_files = []
    for file in files:
        contents = await file.read()
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(class_dir, filename)
        
        # Convert all to RGB JPEG to ensure consistency
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image.save(filepath, "JPEG")
            saved_files.append(filename)
        except Exception:
            # Skip invalid files silently or could raise error
            continue
            
    if not saved_files:
        raise HTTPException(status_code=400, detail="No valid images were uploaded.")
        
    return {"message": f"Successfully uploaded {len(saved_files)} images for class '{class_name}'."}

@app.post("/train")
async def train_model():
    if not os.path.exists(DATASET_DIR):
        raise HTTPException(status_code=400, detail="Dataset directory not found.")
        
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    if len(classes) < 2:
        raise HTTPException(status_code=400, detail="Please upload at least 2 classes to train.")
        
    X = []
    y = []
    
    for class_name in classes:
        class_dir = os.path.join(DATASET_DIR, class_name)
        images = os.listdir(class_dir)
        
        if len(images) == 0:
            raise HTTPException(status_code=400, detail=f"Class '{class_name}' has no images.")
            
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            with open(img_path, "rb") as f:
                try:
                    features = extract_features(f.read())
                    X.append(features.numpy())
                    y.append(class_name)
                except ValueError:
                    continue # skip corrupted
                
    if len(X) == 0:
        raise HTTPException(status_code=400, detail="No valid images found for training.")
        
    # Train Scikit-Learn Logistic Regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    
    # Save Model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
        
    return {"message": "Model trained successfully!", "classes": classes}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Model not trained yet.")
        
    try:
        with open(MODEL_PATH, "rb") as f:
            clf = pickle.load(f)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to load trained model.")
        
    contents = await file.read()
    try:
        features = extract_features(contents).numpy().reshape(1, -1)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Run Inference
    probabilities = clf.predict_proba(features)[0]
    classes = clf.classes_
    
    # Top prediction
    max_idx = probabilities.argmax()
    predicted_class = classes[max_idx]
    confidence = float(probabilities[max_idx] * 100)
    
    # All probabilities for charts
    all_probs = {cls: float(prob * 100) for cls, prob in zip(classes, probabilities)}
    
    return {
        "class": predicted_class,
        "confidence": round(confidence, 2),
        "all_probabilities": all_probs
    }