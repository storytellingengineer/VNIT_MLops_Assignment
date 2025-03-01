# Handwritten Digit Recognition

## 📌 Project Overview
This project focuses on **Handwritten Digit Recognition** using the MNIST dataset. We trained a **Sequential Neural Network** and an **SVC model**, leveraging **MLflow** for tracking experiments. Additionally, we built a **FastAPI** application for model inference and containerized our solution using **Docker**.

This project was assigned by **Dr. Ajay** as part of the **Deployment of ML Models** course.

👨‍💻 **Contributors:** Aayush, Pritesh, Komal, and Poonam.

---

## 🚀 Project Workflow

1. **Data Preparation:**
   - Used the MNIST dataset containing 28x28 grayscale images of handwritten digits (0-9).
   
2. **Model Training:**
   - **Neural Network:** Trained using **Flattened input, Dense layers, and Adam optimizer**.
   - **SVC Model:** Hyperparameter tuning performed using the following grid:
     ```python
     param_grid = {
         'C': [0.1, 1, 10],
         'kernel': ['linear', 'rbf', 'poly'],
         'gamma': ['scale', 'auto']
     }
     ```
   - Both models were tracked using **MLflow**.
   
3. **Model Deployment:**
   - Built a **FastAPI** application to serve the trained model for predictions.
   - Containerized the FastAPI app using **Docker** for easy deployment.

---

## 🔧 Technologies Used

| Component            | Technology Used |
|---------------------|----------------|
| Dataset             | MNIST           |
| Deep Learning Model | Sequential NN (Dense Layers) |
| Optimizer           | Adam            |
| ML Tracking        | MLflow          |
| API Framework       | FastAPI         |
| Containerization    | Docker          |
| Machine Learning Model | SVC (Support Vector Classifier) |

---

## 📊 Experiment Tracking with MLflow
- **MLflow** was used to log:
  - Model parameters
  - Training loss and accuracy
  - SVC hyperparameter tuning results

### Running MLflow UI
To start the MLflow UI and track experiments:
```bash
mlflow ui --host 0.0.0.0 --port 5003
```
Access the MLflow UI at: [http://localhost:5000](http://localhost:5003)

---

## 📡 FastAPI Deployment
We created a FastAPI endpoint to accept handwritten digit images and return predictions from our trained model.

### Example API Request
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"image": "base64-encoded-image"}'
```

### Example API Response
```json
{
  "prediction": 5
}
```

---

## 🐳 Running with Docker
To deploy the API using Docker:
```bash
docker build -t digit_recognition .
docker run -p 8000:8000 digit_recognition
```

## Docker Commands for Building, Tagging, and Pushing the Image

### Build the Docker Image
```sh
docker buildx build -t komalpaliwal266/mlops:assignment .
```
This command builds the Docker image using `buildx` and tags it as `komalpaliwal266/mlops:assignment`.

### Tag the Image
```sh
docker tag f778cfe78466 komalpaliwal266/mlops:assignment
```
Replace `f778cfe78466` with the actual image ID from the previous build step. This command tags the image with the specified repository and tag name.

### Push the Image to Docker Hub
```sh
docker push komalpaliwal266/mlops:assignment
```
This command pushes the tagged image to Docker Hub, making it available for pulling.

### Pull the Image from Docker Hub
```sh
docker pull komalpaliwal266/mlops:assignment
```
This command pulls the image from Docker Hub for use on another system or deployment environment.

### Verify the Image
```sh
docker images | grep mlops
```
This command checks if the image is available locally after pulling or building.

---


## 📌 How to Use
### 1️⃣ Set Up the Environment
If you are running this project in a specific Conda environment:
```bash
conda activate sample-api
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
To create `requirements.txt` file:
```bash
pipreqs .
```
If the above command does not work, use the force parameter:
```bash
pipreqs . --force
```

### 3️⃣ Train the Model
```bash
python train.py
```

### 4️⃣ Start FastAPI Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5️⃣ Create Docker Image and Container
```bash
docker build -t mlops_apis .
```
Or using Buildx:
```bash
docker buildx build -t mlops_apis .
```
To run the container with port mapping:
```bash
docker run -p 5003:5003 mlops_apis
```

### 6️⃣ Run Docker Compose
```bash
docker-compose up --build -d
```

---

## 📌 Future Enhancements
- Implement **CNN** for improved accuracy.
- Deploy the model on a cloud platform.
- Optimize inference performance.

---

## 📩 Contact
If you have any questions, feel free to reach out to us!
