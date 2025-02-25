from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import requests
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from mlxtend.data import loadlocal_mnist
from tensorflow.keras.models import load_model

MLFLOW_TRACKING_URI = "http://localhost:5000/api/2.0/mlflow/runs"

model = load_model("models/tensorflow_model.h5")

app = FastAPI()

class MNISTInput(BaseModel):
    image: list 

@app.post("/predict/")
def predict_digit(data: MNISTInput):
    try:
        input_image = np.array(data.image, dtype=np.float32)

        if input_image.shape != (784,):
            return {"error": "Input must be a list of 784 pixel values."}

        input_image = input_image.reshape(1, 784)  

        predictions = model.predict(input_image)
        predicted_digit = int(np.argmax(predictions))  

        return {"predicted_digit": predicted_digit}

    except Exception as e:
        return {"error": str(e)}

@app.get("/best_model_parameter")
async def get_best_model_parameter():

    response = requests.get(f"{MLFLOW_TRACKING_URI}/search", json={"experiment_ids": ["0"], "max_results": 1})

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"MLflow error: {response.text}")

    runs = response.json().get("runs", [])
    if not runs:
        raise HTTPException(status_code=404, detail="No model parameters found in MLflow.")

    best_params = runs[0]["data"]["params"]
    return {"best_model_parameter": best_params}




@app.post("/training")
async def training(
    images_path: str = './train-images-idx3-ubyte/train-images.idx3-ubyte', 
    labels_path: str = './train-labels-idx1-ubyte/train-labels.idx1-ubyte'
):

    X, y = loadlocal_mnist(
        images_path,
        labels_path
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    # Scale data using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }

    # Perform grid search with cross-validation to find best SVM model
    svm_model = svm.SVC()
    grid_search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Get best SVM model and its metrics
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Create JSON payload for MLflow API request
    payload = {
        "experiment_id": "0",
        "run_name": "SVM Model Training",
        "params": best_params,
        "metrics": {"best_score": best_score},
        "tags": {"model": "svm"}
    }

    response = requests.post(f"{MLFLOW_TRACKING_URI}/create", json=payload)

    if response.status_code == 200:
        return {"message": "Training completed and run logged successfully!", "best_params": best_params}
    else:
        return {"error": f"MLflow API error: {response.text}"}

# Run the API using:
# uvicorn filename:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003, reload=True)
