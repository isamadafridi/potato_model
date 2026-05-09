from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

modle_path = 'potato_model_mobileNetV2_v1.keras'
model = tf.keras.models.load_model(modle_path)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/")
def home():
    return {"message": "Potato Disease API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents  = await file.read()
    image     = Image.open(io.BytesIO(contents)).convert("RGB")
    image     = image.resize((256, 256))

    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)

    predictions     = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence      = float(np.max(predictions[0])) * 100

    print(f"Early Blight : {predictions[0][0]*100:.2f}%")
    print(f"Late Blight  : {predictions[0][1]*100:.2f}%")
    print(f"Healthy      : {predictions[0][2]*100:.2f}%")
    print(f"Predicted    : {predicted_class} ({confidence:.2f}%)")

    return {
        "class"      : predicted_class,
        "confidence" : f"{confidence:.2f}%"
    }