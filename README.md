# 🌸 Flower Classification API with Transfer Learning

Bu proje, çiçek görsellerini sınıflandırmak için geliştirilmiş yapay zeka tabanlı bir görüntü sınıflandırma sistemidir. Model, TensorFlow Flower Dataset kullanılarak eğitilmiş ve Google Cloud Run üzerinde canlı bir REST API olarak deploy edilmiştir.

## 📌 Proje Amacı

Bu projenin amacı, derin öğrenme ve bulut bilişim teknolojilerini bir araya getirerek çiçek görsellerini otomatik olarak sınıflandıran bir sistem geliştirmektir.

Model, verilen bir çiçek görselini aşağıdaki sınıflardan birine tahmin eder:

- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

## 🚀 Kullanılan Teknolojiler

- Python
- TensorFlow / Keras
- MobileNetV2
- Transfer Learning
- Fine-Tuning
- Flask
- Docker
- Google Cloud Run
- Google Cloud Build
- Google Artifact Registry
- Postman

## 🧠 Model Yapısı

Projede önceden eğitilmiş MobileNetV2 modeli kullanılmıştır. MobileNetV2, ImageNet veri seti üzerinde eğitilmiş hafif ve hızlı bir CNN modelidir.

Model geliştirme sürecinde:

- Görseller 224x224 boyutuna getirildi.
- Veri seti eğitim ve doğrulama olarak ayrıldı.
- Data augmentation uygulandı.
- İlk aşamada MobileNetV2 katmanları donduruldu.
- Son aşamada fine-tuning ile son katmanlar tekrar eğitildi.

## 📊 Model Performansı

Fine-tuning sonrası modelin genel doğruluk oranı yaklaşık olarak:

```text
Accuracy: 0.91
Weighted F1-score: 0.91
```

Classification Report:

```text
              precision    recall  f1-score   support

       daisy       0.88      0.95      0.91       107
   dandelion       0.95      0.95      0.95       191
       roses       0.85      0.87      0.86       119
  sunflowers       0.92      0.90      0.91       135
      tulips       0.90      0.87      0.89       182

    accuracy                           0.91       734
   macro avg       0.90      0.91      0.90       734
weighted avg       0.91      0.91      0.91       734
```

## 📁 Proje Dosya Yapısı

```text
bulut-proje/
│
├── app.py
├── train_model.py
├── requirements.txt
├── Dockerfile
├── runtime.txt
├── README.md
│
├── dataset/
│   └── flower_photos/
│
├── models/
│   ├── flower_model.keras
│   ├── accuracy_graph.png
│   ├── loss_graph.png
│   └── confusion_matrix.png
│
├── uploads/
└── utils/
```

## ⚙️ Kurulum

Projeyi klonlayın:

```bash
git clone https://github.com/ebraar/bulut-proje.git
cd bulut-proje
```

Sanal ortam oluşturun:

```bash
python -m venv venv
source venv/bin/activate
```

Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

## 🏋️ Model Eğitimi

Modeli eğitmek için:

```bash
python train_model.py
```

Eğitim sonunda model şu konuma kaydedilir:

```text
models/flower_model.keras
```

Ayrıca accuracy, loss ve confusion matrix grafikleri de `models` klasörüne kaydedilir.

## 🌐 Flask API Çalıştırma

Local ortamda API’yi çalıştırmak için:

```bash
python app.py
```

API varsayılan olarak şu adreste çalışır:

```text
http://127.0.0.1:8080
```

## 🔗 API Endpointleri

### Ana Endpoint

```http
GET /
```

API’nin çalışıp çalışmadığını kontrol eder.

Örnek response:

```json
{
  "message": "Flower Classification API is running"
}
```

### Görsel Sınıflandırma Endpointi

```http
POST /classify
```

Bir çiçek görseli alır ve sınıf tahmini yapar.

Postman kullanımı:

- Method: `POST`
- Body: `form-data`
- Key: `file`
- Type: `File`

Örnek response:

```json
{
  "predicted_class": "sunflowers",
  "confidence": 0.92,
  "all_scores": {
    "daisy": 0.01,
    "dandelion": 0.03,
    "roses": 0.02,
    "sunflowers": 0.92,
    "tulips": 0.02
  }
}
```

## 🐳 Docker Kullanımı

Docker image oluşturmak için:

```bash
docker build -t flower-classification-api .
```

Container çalıştırmak için:

```bash
docker run -p 8080:8080 flower-classification-api
```

## ☁️ Google Cloud Deployment

Proje Google Cloud Run üzerinde deploy edilmiştir.

Kullanılan servisler:

- Cloud Run
- Cloud Build
- Artifact Registry

Docker image build işlemi:

```bash
gcloud builds submit \
  --tag europe-west1-docker.pkg.dev/bulut-proje-496715/bulut-proje/bulut-proje:latest
```

Cloud Run deploy işlemi:

```bash
gcloud run deploy bulut-proje \
  --image europe-west1-docker.pkg.dev/bulut-proje-496715/bulut-proje/bulut-proje:latest \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300
```

## 🌍 Canlı API

Proje Google Cloud Run üzerinde canlı olarak çalışmaktadır:

```text
https://bulut-proje-637925721962.europe-west1.run.app
```

Canlı sınıflandırma endpointi:

```text
https://bulut-proje-637925721962.europe-west1.run.app/classify
```

## 🔍 Karşılaşılan Problemler ve Çözümler

### 1. Cloud Run Port Problemi

Cloud Run uygulamanın `8080` portunu dinlemesini beklediği için Flask uygulaması şu şekilde düzenlendi:

```python
port = int(os.environ.get("PORT", 8080))
app.run(host="0.0.0.0", port=port)
```

### 2. Model Startup Problemi

Model başlangıçta yüklendiğinde container geç açıldığı için lazy loading uygulandı. Model ilk istek geldiğinde yüklenmektedir.

### 3. Yanlış Tahmin Problemi

Başlangıçta görsel iki kere normalize edildiği için bazı tahminler hatalı çıkıyordu. Modelin içinde zaten `Rescaling` katmanı olduğu için API tarafındaki ekstra normalization kaldırıldı.

## 📌 Sonuç

Bu proje ile:

- CNN tabanlı görüntü sınıflandırma modeli geliştirildi.
- Transfer Learning ve Fine-Tuning uygulandı.
- Model Flask API’ye dönüştürüldü.
- Docker ile containerlaştırıldı.
- Google Cloud Run üzerinde canlı olarak deploy edildi.
- Postman ile canlı inference testi yapıldı.

## 👩‍💻 Geliştirici

**Ebrar Betül Akgül**