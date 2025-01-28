from fastapi import FastAPI
import onnxruntime
import numpy as np

app = FastAPI()

# Load the ONNX model
session = onnxruntime.InferenceSession("model.onnx")

@app.post("/predict/")
def predict(data: dict):
    # Convert input features to numpy array
    input_data = np.array([data['features']], dtype=np.float32)
    
    # Get the input name for the ONNX model
    input_name = session.get_inputs()[0].name
    
    # Run inference
    prediction = session.run(None, {input_name: input_data})
    
    return {"prediction": prediction[0].tolist()}
