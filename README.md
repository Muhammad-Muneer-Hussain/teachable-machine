# Offline Teachable Machine 🤖

A full-stack, offline-first replication of Google's Teachable Machine built using **FastAPI** for the backend and **Streamlit** for the frontend. This application allows users to train functional machine learning models directly via their browser using image uploads or a live webcam feed—completely containerized and local.

---

## 📺 Project Demo & Visuals

```markdown
### Live Interface Overview
![App Screenshot](media\app screen shot.PNG)

### Video / GIF Demo
![Working Demo](media\project working video.mp4)


## ✨ Features

* **Offline-First Architecture:** No cloud dependencies or external API calls, ensuring complete data privacy.
* **Dual Data Collection:** Upload custom image datasets from local storage or capture samples using a real-time webcam stream.
* **Transfer Learning Backend:** Utilizes a pre-trained **MobileNetV3** (Small) neural network as a feature extractor coupled with a **Logistic Regression** head for lightning-fast training.
* **Interactive Frontend:** Built with Streamlit components and embedded JavaScript for smooth webcam handling and live confidence score metrics.
* **Containerized Deployment:** Microservices architecture configured natively with Docker and Docker Compose.

---

## 📁 Project Directory Structure

```text
TEACHABLE-MACHINE/
├── backend/
│   ├── main.py                # FastAPI implementation & ML pipeline
│   └── Dockerfile             # Container configuration for backend
├── frontend/
│   ├── app.py                 # Streamlit UI & JS Component handling
│   └── Dockerfile             # Container configuration for frontend
├── dataset/                   # Local volume mapping for uploaded samples (Auto-generated)
├── models/                    # Local volume mapping for trained pickle files (Auto-generated)
├── docker-compose.yml         # Multi-container orchestration config
└── requirements.txt           # Unified python ecosystem dependencies

🚀 Getting Started
Prerequisites
Make sure you have the following software utilities installed on your host environment:

Docker (Desktop or Engine)

Docker Compose

🛠️ Execution & Deployment Steps
Follow these basic commands to initialize the ecosystem from the root folder:

Build and Run the Containers:
docker-compose up --build

Accessing the Interfaces:

Frontend UI (Streamlit): Open your web browser and navigate to http://localhost:8501

Backend API Documentation (FastAPI Docs): View the interactive Swagger documentation at http://localhost:8000/docs

Stopping the Services:
docker-compose down

🧠 Technical Workflow
1. Feature Extraction:
 When an image frame is uploaded or snapped via the webcam layout, it is preprocessed into a $224 \times 224$ tensor and passed into MobileNet_V3_Small. The final classification head is neutralized (nn.Identity()) to capture generic embeddings/feature maps.
 2. Fast Training Head: Scikit-Learn's LogisticRegression takes the multi-dimensional embeddings alongside the class labels to optimize and fit the decision boundary within seconds.
 3. Serialized Predictions: The state of the weights is exported locally into a serialized binary format (model.pkl), which is shared interactively across endpoints for real-time validation matrices.
 🛠️ Technology Stack Used
Frontend UI: Streamlit, HTML5/JavaScript Canvas (Webcam Stream)

Backend Framework: FastAPI, Uvicorn

Deep Learning Framework: PyTorch (torch), Torchvision

Machine Learning Algorithms: Scikit-Learn (Logistic Regression)

Image Processing Engine: Pillow (PIL)

Container Environment: Docker & Docker Compose