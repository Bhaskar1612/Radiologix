# Radiologix
Radiologix is a medical imaging analysis project that uses deep learning models to classify various diseases based on radiology X-rays. The project leverages FastAPI as the backend framework and React for the frontend interface. The backend integrates multiple PyTorch models to provide accurate and fast disease classification, making it useful for healthcare professionals to aid in diagnosis.

## Key Features

Backend Framework: Built with FastAPI for high performance and ease of use.
Frontend Interface: Developed using React to provide a user-friendly interface for uploading images and displaying results.
Machine Learning Models: Four deep learning models developed using PyTorch and TorchVision:

- Pneumonia Model: Classifies X-rays into 'NORMAL' or 'PNEUMONIA'. (Accuracy: 81.5%)
- Brain MRI Model: Classifies MRI scans into 'Glioma', 'Meningioma', 'No Tumor', or 'Pituitary'. (Accuracy: 78%)
- Lumbar Spine Model: Classifies lumbar spine X-rays into 'processed_lsd', 'processed_osf', 'processed_spider', or 'processed_tseg'. (Accuracy: 98%)
- COVID-19 Model: Classifies X-rays into 'Covid', 'Normal', or 'Viral Pneumonia'. (Accuracy: 88%)

DevOps Integration: The project uses Docker for containerization, Kubernetes for orchestration, and a CI/CD pipeline maintained through Git workflows.

## Installation and Setup
Prerequisites
- Python 3.8 or higher
- Node.js and npm (for React)
- Docker
- Kubernetes
- Backend Setup (FastAPI)

Clone the Repository:
bash
git clone <repository-url>
cd radiologix/backend

Build Docker Image:
bash
docker build -t radiologix-backend .

Run Docker Container:
bash
docker run -p 8000:8000 radiologix-backend

Kubernetes Deployment:
Deploy the backend using the kubernetes-backend.yaml file:
bash
kubectl apply -f kubernetes-backend.yaml

Frontend Setup (React)

Navigate to Frontend Directory:
bash
cd radiologix/frontend

Build Docker Image:
bash
docker build -t radiologix-frontend .

Run Docker Container:
bash
docker run -p 3000:3000 radiologix-frontend

Kubernetes Deployment:
Deploy the frontend using the kubernetes-frontend.yaml file:
bash
kubectl apply -f kubernetes-frontend.yaml

## CI/CD Pipeline
The project is maintained using a Git-based CI/CD workflow, ensuring that every commit is tested and deployed automatically:

Continuous Integration: On every push, the backend and frontend Docker images are built and tested.
Continuous Deployment: Successful builds are deployed to a Kubernetes cluster.
Usage
Open the frontend in a web browser.
Upload an X-ray image for classification.
The frontend sends the image to the FastAPI backend.
The backend loads the appropriate PyTorch model and returns the classification result.
The result is displayed on the frontend.
Real-Life Importance
Radiologix is designed to assist healthcare professionals in early and accurate diagnosis of various diseases, including Pneumonia, brain tumors, lumbar spine issues, and COVID-19. The use of AI in medical imaging can significantly reduce diagnostic errors, expedite patient care, and enhance decision-making processes.

## Potential Benefits:
Early Detection: Quickly identifies conditions like pneumonia and brain tumors, allowing for timely intervention.
Cost-Effective: Reduces the need for multiple tests by providing accurate results from a single X-ray.
Scalable: Can be integrated into hospital management systems to provide instant diagnostic support across various departments.
Automated Workflow: With DevOps practices, the deployment and management of this tool are fully automated, ensuring high availability and minimal downtime.
Future Enhancements
Support for additional disease classifications using different imaging modalities.
Improved accuracy and performance of models through continuous training on larger datasets.
Integration with Electronic Health Record (EHR) systems for seamless medical data management.
License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
FastAPI
React
PyTorch
