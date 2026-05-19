from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Upload klasörü
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Lazy loading için model başlangıçta None
model = None

# Sınıf isimleri
class_names = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
]

IMG_SIZE = (224, 224)


@app.route("/")
def home():
    return jsonify({
        "message": "Flower Classification API is running"
    })


@app.route("/classify", methods=["POST"])
def classify_image():
    global model

    if model is None:
        model = tf.keras.models.load_model("models/flower_model.keras")

    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "Empty filename"
        }), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = Image.open(filepath).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Burada ekstra normalize etmiyoruz.
    # Çünkü model içinde layers.Rescaling(1./127.5, offset=-1) zaten var.
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions[0]))

    all_scores = {}

    for i, class_name in enumerate(class_names):
        all_scores[class_name] = float(predictions[0][i])

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_scores": all_scores
    })


@app.route("/dataset/upload", methods=["POST"])
def upload_dataset():
    return jsonify({
        "message": "Dataset upload endpoint hazır"
    })


@app.route("/model/finetune", methods=["POST"])
def finetune_model():
    return jsonify({
        "message": "Fine-tune endpoint hazır"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)