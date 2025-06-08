from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json

app = FastAPI()

# Load class names from JSON file
try:
    with open("class_names (1).json", "r") as f:
        CLASS_NAMES = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load class names from JSON: {e}")

# Load model
try:
    MODEL = tf.keras.models.load_model("../saved_models/base_model.keras")
    output_shape = MODEL.output_shape[-1]
    assert output_shape == len(CLASS_NAMES), \
        f"Model output size ({output_shape}) does not match number of class names ({len(CLASS_NAMES)})"

    print("\n=== MODEL ARCHITECTURE ===")
    MODEL.summary()
    print(f"\nInput shape: {MODEL.input_shape}")
    print(f"Output shape: {MODEL.output_shape}")
except Exception as e:
    print(f"\n!!! ERROR LOADING MODEL: {e}")
    raise

@app.get("/model-info")
async def model_info():
    return {
        "input_shape": MODEL.input_shape,
        "output_shape": MODEL.output_shape,
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES
    }

def preprocess_image(data):
    try:
        image = Image.open(BytesIO(data))
        image = image.convert("RGB")
        image = image.resize((128, 128))  # Match model input
        image_array = np.array(image).astype("float32") / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        print(f"Image processing error: {e}")
        raise

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = preprocess_image(image_data)

        predictions = MODEL.predict(image)
        prediction_vector = predictions[0]
        predicted_index = int(np.argmax(prediction_vector))

        # Safety check
        if not (0 <= predicted_index < len(CLASS_NAMES)):
            raise ValueError(f"Predicted index {predicted_index} is out of range for class list of length {len(CLASS_NAMES)}")

        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(prediction_vector[predicted_index])

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": {name: float(prob) for name, prob in zip(CLASS_NAMES, prediction_vector)}
        }

    except Exception as e:
        print(f"\nPrediction error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("\nStarting server with debug mode...")
    uvicorn.run(app, host="localhost", port=8000, log_level="debug")
