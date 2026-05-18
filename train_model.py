import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# Dataset yolu
dataset_path = "dataset/flower_photos"

# Görsel boyutu
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Dataset yükleme
train_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Sınıf isimleri
class_names = train_dataset.class_names
print("Classes:", class_names)

# Performans optimizasyonu
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# Önceden eğitilmiş MobileNetV2 modeli
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Transfer learning: temel modeli donduruyoruz
base_model.trainable = False

# Model oluşturma
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1. / 127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation="softmax")
])

# Model derleme
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Early stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Eğitim
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    callbacks=[early_stopping]
)

# Models klasörü oluştur
os.makedirs("models", exist_ok=True)

# Model kaydet
model.save("models/flower_model.keras")
print("Model kaydedildi: models/flower_model.keras")


# Accuracy grafiği
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

plt.figure(figsize=(8, 6))
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend()
plt.title("Accuracy Graph")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("models/accuracy_graph.png")
plt.show()


# Loss grafiği
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8, 6))
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend()
plt.title("Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("models/loss_graph.png")
plt.show()


# Confusion Matrix için gerçek ve tahmin değerleri
y_true = []
y_pred = []

for images, labels in validation_dataset:
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.savefig("models/confusion_matrix.png")
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))