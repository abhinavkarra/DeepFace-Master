# DeepFace - Advanced Face Recognition & Analysis Framework

<div align="center">

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/abhinavkarra/DeepFace-Master?style=social)](https://github.com/abhinavkarra/DeepFace-Master/stargazers)

<p align="center">
  <img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-icon-labeled.png" width="200" height="240">
</p>

**A lightweight, production-ready face recognition and facial attribute analysis framework for Python**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [API](#-api-reference) • [Models](#-supported-models)

</div>

---

## 🌟 Features

### Face Recognition & Verification
- **11 State-of-the-art Models**: VGG-Face, FaceNet, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace, GhostFaceNet, Buffalo_L, QMAG
- **High Accuracy**: Exceeds human-level performance (97.53%) on facial recognition tasks
- **Fast Processing**: Optimized for both CPU and GPU environments

### Face Detection
- **10+ Detector Options**: OpenCV, SSD, MTCNN, RetinaFace, MediaPipe, Dlib, YOLO, YuNet, CenterFace, FastMTCNN
- **Robust Detection**: Works with various angles, lighting conditions, and occlusions
- **Face Alignment**: Automatic alignment for improved recognition accuracy

### Demographic Analysis
- **Age Prediction**: Apparent age estimation
- **Gender Recognition**: Male/Female classification
- **Emotion Detection**: 7 emotions (happy, sad, angry, surprise, fear, disgust, neutral)
- **Race Classification**: Ethnicity prediction across multiple categories

### Anti-Spoofing
- **Liveness Detection**: FasNet-based spoofing detection
- **Security**: Protection against photo and video attacks

### Database Integration
- **Vector Databases**: Weaviate support for scalable face search
- **SQL Databases**: PostgreSQL and MongoDB integration
- **Caching**: Intelligent caching for faster repeated lookups

### Production-Ready
- **REST API**: Flask-based API for easy integration
- **Real-time Processing**: Webcam streaming support
- **Encryption**: Built-in homomorphic encryption for privacy
- **Comprehensive Docs**: Architecture diagrams and detailed pipeline documentation

---

## 📦 Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### From Source

```bash
git clone https://github.com/abhinavkarra/DeepFace-Master.git
cd DeepFace-Master
pip install -e .
```

### Requirements

- Python 3.7+
- TensorFlow 1.9.0+
- Keras 2.2.0+
- OpenCV 4.5.5+
- NumPy, Pandas, Pillow

---

## 🚀 Quick Start

### Face Verification

Verify if two images contain the same person:

```python
from deepface import DeepFace

result = DeepFace.verify(
    img1_path="person1.jpg",
    img2_path="person2.jpg",
    model_name="VGG-Face",
    detector_backend="retinaface"
)

print(f"Same person: {result['verified']}")
print(f"Distance: {result['distance']}")
```

### Face Recognition

Find a person in a database:

```python
from deepface import DeepFace

results = DeepFace.find(
    img_path="target.jpg",
    db_path="face_database/",
    model_name="ArcFace",
    detector_backend="retinaface"
)

print(results)
```

### Facial Attribute Analysis

Analyze age, gender, emotion, and race:

```python
from deepface import DeepFace

analysis = DeepFace.analyze(
    img_path="person.jpg",
    actions=["age", "gender", "emotion", "race"],
    detector_backend="retinaface"
)

print(f"Age: {analysis['age']}")
print(f"Gender: {analysis['dominant_gender']}")
print(f"Emotion: {analysis['dominant_emotion']}")
print(f"Race: {analysis['dominant_race']}")
```

### Face Embeddings

Generate face embeddings/representations:

```python
from deepface import DeepFace

embedding = DeepFace.represent(
    img_path="person.jpg",
    model_name="Facenet",
    detector_backend="retinaface"
)

print(f"Embedding dimension: {len(embedding)}")
```

### Real-time Face Detection

Process webcam stream with face detection and analysis:

```python
from deepface import DeepFace

DeepFace.stream(
    db_path="face_database/",
    model_name="VGG-Face",
    detector_backend="retinaface",
    time_threshold=5,
    frame_threshold=3
)
```

---

## 🎯 Supported Models

### Recognition Models

| Model | Output Dimension | Performance | Speed |
|-------|-----------------|-------------|-------|
| **VGG-Face** | 2,622 | High | Medium |
| **Facenet** | 128 | Very High | Fast |
| **Facenet512** | 512 | Very High | Fast |
| **OpenFace** | 128 | Good | Very Fast |
| **DeepFace** | 4,096 | High | Slow |
| **DeepID** | 160 | Good | Fast |
| **ArcFace** | 512 | Very High | Fast |
| **Dlib** | 128 | Good | Fast |
| **SFace** | 128 | High | Fast |
| **GhostFaceNet** | 512 | High | Very Fast |
| **Buffalo_L** | 512 | Very High | Fast |
| **QMAG** | Custom | High | Fast |

### Detection Backends

| Detector | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| **OpenCV** | Very Fast | Good | General purpose |
| **SSD** | Fast | Good | Balanced |
| **MTCNN** | Medium | High | High accuracy |
| **RetinaFace** | Medium | Very High | Best accuracy |
| **MediaPipe** | Very Fast | Good | Real-time |
| **Dlib** | Fast | Good | Lightweight |
| **YOLO** | Fast | High | Modern detection |
| **YuNet** | Fast | Good | OpenCV optimized |
| **CenterFace** | Fast | High | Anchor-free |

---

## 🔧 API Reference

### REST API

Start the Flask API server:

```bash
cd deepface/api/src
python app.py
```

The API will be available at `http://localhost:5000`

#### Endpoints

**Verify**: `POST /verify`
```json
{
  "img1_path": "path/to/image1.jpg",
  "img2_path": "path/to/image2.jpg",
  "model_name": "VGG-Face",
  "detector_backend": "retinaface"
}
```

**Analyze**: `POST /analyze`
```json
{
  "img_path": "path/to/image.jpg",
  "actions": ["age", "gender", "emotion", "race"],
  "detector_backend": "retinaface"
}
```

**Find**: `POST /find`
```json
{
  "img_path": "path/to/image.jpg",
  "db_path": "face_database/",
  "model_name": "ArcFace"
}
```

---

## 📊 Architecture

DeepFace follows a modular architecture with clear separation of concerns:

```
DeepFace
├── Detection Module      → Face detection and alignment
├── Recognition Module    → Face embeddings and matching
├── Verification Module   → Face verification pipeline
├── Analysis Module       → Demographic attribute analysis
├── Streaming Module      → Real-time processing
└── Database Module       → Vector and SQL database integration
```

For detailed architecture diagrams, see [ARCHITECTURE.md](ARCHITECTURE.md)

For pipeline details, see [DETAILED_PIPELINE.md](DETAILED_PIPELINE.md)

---

## 🗄️ Database Support

### Vector Database (Weaviate)

```python
from deepface.modules.database import weaviate

db = weaviate.WeaviateClient(
    url="http://localhost:8080",
    api_key="your-api-key"
)

# Store embeddings
db.store(embedding, metadata)

# Search similar faces
results = db.search(query_embedding, limit=10)
```

### SQL Database (PostgreSQL)

```python
from deepface.modules.database import postgres

db = postgres.PostgresClient(
    host="localhost",
    database="deepface",
    user="user",
    password="password"
)
```

---

## 🎥 Webcam Demo

Run the webcam demo:

```bash
python deepface_webcam.py
```

Or use the specialized QMAG detector:

```bash
python scripts/webcam_qmag.py
```

---

## 📁 Project Structure

```
deepface-master/
├── deepface/                    # Main package
│   ├── DeepFace.py             # Main API interface
│   ├── commons/                # Utility functions
│   ├── config/                 # Configuration files
│   ├── models/                 # Model implementations
│   │   ├── facial_recognition/ # Recognition models
│   │   ├── face_detection/     # Detection models
│   │   ├── demography/         # Age, gender, emotion, race
│   │   └── spoofing/           # Anti-spoofing models
│   ├── modules/                # Core functionality
│   │   ├── detection.py
│   │   ├── recognition.py
│   │   ├── verification.py
│   │   ├── demography.py
│   │   ├── streaming.py
│   │   └── database/
│   └── api/                    # REST API
│       └── src/
├── face_database/              # Sample database
├── scripts/                    # Utility scripts
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

---

## 🔒 Privacy & Security

- **Encryption**: Homomorphic encryption support for privacy-preserving face recognition
- **Local Processing**: All computations can run locally without external API calls
- **No Data Collection**: No telemetry or data collection
- **Open Source**: Fully auditable codebase

---

## 📈 Performance

### Accuracy Benchmarks

See [ACCURACY_ANALYSIS.md](ACCURACY_ANALYSIS.md) for detailed accuracy metrics across different models and datasets.

### Speed Benchmarks

| Model | CPU (ms) | GPU (ms) |
|-------|----------|----------|
| VGG-Face | 250 | 25 |
| Facenet | 150 | 15 |
| ArcFace | 180 | 18 |
| OpenFace | 120 | 12 |

*Benchmarks on Intel i7-10700K and NVIDIA RTX 3070*

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Based on the original [DeepFace framework](https://github.com/serengil/deepface) by Sefik Ilkin Serengil
- Face detection models from various research papers and implementations
- Recognition models from state-of-the-art research in face recognition

---

## 📧 Contact

For questions, issues, or suggestions:

- **GitHub Issues**: [Create an issue](https://github.com/abhinavkarra/DeepFace-Master/issues)
- **Repository**: [DeepFace-Master](https://github.com/abhinavkarra/DeepFace-Master)

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star ⭐

---

<div align="center">

Made with ❤️ for the Computer Vision Community

</div>
