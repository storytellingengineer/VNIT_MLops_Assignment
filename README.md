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

---

## 📌 How to Use
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Train the Model
```bash
python train.py
```
### 4️⃣ Start FastAPI Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 📌 Future Enhancements
- Implement **CNN** for improved accuracy.
- Deploy the model on a cloud platform.
- Optimize inference performance.

---

## 📩 Contact
If you have any questions, feel free to reach out to us!
