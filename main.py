from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()

# Завантаження TFLite-моделі
interpreter = tf.lite.Interpreter(model_path="frozen_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Завантаження назв класів
def unpickle_class_names():
    with open('data/meta', 'rb') as f:
        meta_data = pickle.load(f, encoding='bytes')
    fine_class_names = [name.decode('utf-8') for name in meta_data[b'fine_label_names']]
    return fine_class_names

fine_names = unpickle_class_names()

# Функція передобробки зображення
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((32, 32))  # CIFAR-100 формат
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_data = preprocess_image(image_bytes)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = int(np.argmax(output_data))
    confidence = float(np.max(output_data))
    predicted_label = fine_names[predicted_class]

    return JSONResponse({
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "confidence": confidence
    })
