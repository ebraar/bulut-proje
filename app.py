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

# Model yükleme
model = tf.keras.models.load_model("models/flower_model.keras")

# Sınıf isimleri
class_names = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
]

IMG_SIZE = (224, 224)


# Ana endpoint
@app.route("/")
def home():
    return jsonify({
        "message": "Flower Classification API is running"
    })


# Görsel sınıflandırma endpointi
@app.route("/classify", methods=["POST"])
def classify_image():

    # Dosya kontrolü
    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]

    # Dosya adı kontrolü
    if file.filename == "":
        return jsonify({
            "error": "Empty filename"
        }), 400

    # Dosya yolu
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)

    # Dosyayı kaydet
    file.save(filepath)

    # Görseli yükle
    img = Image.open(filepath).convert("RGB")
    img = img.resize(IMG_SIZE)

    # NumPy array'e çevir
    img_array = image.img_to_array(img)

    # Batch dimension ekle
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize et
    img_array = (img_array / 127.5) - 1

    # Tahmin yap
    predictions = model.predict(img_array)

    # En yüksek skorlu sınıf
    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = float(np.max(predictions[0]))

    # Tüm skorlar
    all_scores = {}

    for i, class_name in enumerate(class_names):
        all_scores[class_name] = float(predictions[0][i])

    # JSON response
    return jsonify({
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_scores": all_scores
    })


# Dataset upload endpoint
@app.route("/dataset/upload", methods=["POST"])
def upload_dataset():

    return jsonify({
        "message": "Dataset upload endpoint hazır"
    })


# Fine-tune endpoint
@app.route("/model/finetune", methods=["POST"])
def finetune_model():

    return jsonify({
        "message": "Fine-tune endpoint hazır"
    })


# Uygulama çalıştırma
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)