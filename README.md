# Offline Teachable Machine 🤖

A full-stack, offline-first replication of Google's Teachable Machine built using **FastAPI** for the backend and **Streamlit** for the frontend. This application allows users to train functional machine learning models directly via their browser using image uploads or a live webcam feed—completely containerized and local.

---

## 📺 Project Demo & Visuals

### Live Interface Overview
![App Screenshot](frontend/models/model.pkl) > **Note:** Below are placeholders to add your project's working screenshots or a quick video/GIF demo.
> 
> * **Training Interface:** Put a picture of sample gathering here.
> * **Real-time Prediction:** Put a picture of successful classification here.

---

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