from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model (ensure "tensorflow_model.h5" exists)
model = load_model("models/tensorflow_model.h5")

# Initialize FastAPI
app = FastAPI()

# Define input data format
class MNISTInput(BaseModel):
    image: list  # List of 784 pixel values

@app.post("/predict/")
def predict_digit(data: MNISTInput):
    try:
        # Convert input list to NumPy array
        input_image = np.array(data.image, dtype=np.float32)

        # Validate input shape (should be 784 pixels)
        if input_image.shape != (784,):
            return {"error": "Input must be a list of 784 pixel values."}

        # Reshape input for model
        input_image = input_image.reshape(1, 784)  # Reshape to (1, 784)

        # Make a prediction
        predictions = model.predict(input_image)
        predicted_digit = int(np.argmax(predictions))  # Get highest probability class

        return {"predicted_digit": predicted_digit}

    except Exception as e:
        return {"error": str(e)}

# Run the API using:
# uvicorn filename:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003, reload=True)