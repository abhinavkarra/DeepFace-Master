# DeepFace

> **Production-ready face recognition & analysis API for Python**  
> 11 recognition models • 10+ detectors • Real-time streaming • Database integration

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/abhinavkarra/DeepFace-Master/pulls)

---

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from deepface import DeepFace

# Verify two faces
result = DeepFace.verify("img1.jpg", "img2.jpg")
print(result['verified'])  # True/False

# Find face in database
matches = DeepFace.find("target.jpg", db_path="database/")

# Analyze demographics
attrs = DeepFace.analyze("person.jpg", actions=['age', 'gender', 'emotion'])
print(f"{attrs['age']} years old, {attrs['dominant_emotion']}")
```

---

## API Reference

### Core Functions

#### `DeepFace.verify()`
**Compare two face images**

```python
result = DeepFace.verify(
    img1_path="person1.jpg",
    img2_path="person2.jpg",
    model_name="VGG-Face",      # VGG-Face, Facenet, OpenFace, DeepFace, DeepID, 
                                 # QMAG, Dlib, SFace, GhostFaceNet, Buffalo_L
    detector_backend="retinaface", # opencv, ssd, mtcnn, retinaface, mediapipe, 
                                   # dlib, yolo, yunet, centerface
    distance_metric="cosine",    # cosine, euclidean, euclidean_l2
    enforce_detection=True,      # Raise error if no face detected
    align=True,                  # Align faces before recognition
    normalization="base"         # base, raw, Facenet, VGGFace, QMAG
)

# Returns:
# {
#   'verified': bool,
#   'distance': float,
#   'threshold': float,
#   'model': str,
#   'detector_backend': str,
#   'similarity_metric': str,
#   'facial_areas': {...},
#   'time': float
# }
```

#### `DeepFace.find()`
**Search face in image database**

```python
results = DeepFace.find(
    img_path="target.jpg",
    db_path="face_database/",
    model_name="QMAG",
    detector_backend="retinaface",
    distance_metric="cosine",
    enforce_detection=True,
    align=True,
    normalization="base",
    silent=False
)

# Returns: pandas DataFrame with columns:
# - identity: matched image path
# - distance: similarity distance
# - threshold: model threshold
# All images within threshold are returned
```

#### `DeepFace.analyze()`
**Facial attribute analysis**

```python
analysis = DeepFace.analyze(
    img_path="person.jpg",
    actions=['age', 'gender', 'emotion', 'race'],
    detector_backend="retinaface",
    enforce_detection=True,
    align=True,
    silent=False
)

# Returns:
# {
#   'age': int,
#   'gender': {'Man': float, 'Woman': float},
#   'dominant_gender': str,
#   'emotion': {'angry': float, 'disgust': float, 'fear': float, 
#               'happy': float, 'sad': float, 'surprise': float, 'neutral': float},
#   'dominant_emotion': str,
#   'race': {'asian': float, 'indian': float, 'black': float, 
#            'white': float, 'middle eastern': float, 'latino hispanic': float},
#   'dominant_race': str,
#   'region': {'x': int, 'y': int, 'w': int, 'h': int}
# }
```

#### `DeepFace.represent()`
**Generate face embeddings/vectors**

```python
embedding = DeepFace.represent(
    img_path="face.jpg",
    model_name="Facenet",
    detector_backend="retinaface",
    enforce_detection=True,
    align=True,
    normalization="base"
)

# Returns: list of float (embedding vector)
# Length depends on model:
# - VGG-Face: 2622
# - Facenet: 128
# - Facenet512: 512
# - OpenFace: 128
# - DeepFace: 4096
# - DeepID: 160
# - QMAG: 512
# - Dlib: 128
# - SFace: 128
# - GhostFaceNet: 512
# - Buffalo_L: 512
```

#### `DeepFace.stream()`
**Real-time face recognition from webcam**

```python
DeepFace.stream(
    db_path="face_database/",
    model_name="VGG-Face",
    detector_backend="opencv",
    distance_metric="cosine",
    enable_face_analysis=True,
    source=0,                    # 0 for default webcam
    time_threshold=5,            # Show same person every 5 seconds
    frame_threshold=3            # Process every 3rd frame
)
```

#### `DeepFace.extract_faces()`
**Detect and extract all faces from image**

```python
faces = DeepFace.extract_faces(
    img_path="group.jpg",
    detector_backend="retinaface",
    enforce_detection=True,
    align=True,
    grayscale=False
)

# Returns: list of dicts with:
# - face: numpy array (cropped face)
# - facial_area: {x, y, w, h}
# - confidence: detection confidence
```

---

## Model Performance

### Recognition Models

| Model | Dimension | LFW Accuracy | Speed | Best For |
|-------|-----------|--------------|-------|----------|
| **VGG-Face** | 2,622 | 98.95% | Medium | General purpose |
| **Facenet** | 128 | 99.65% | Fast | Production systems |
| **Facenet512** | 512 | 99.65% | Fast | High accuracy |
| **QMAG** | 512 | 99.82% | Fast | State-of-the-art |
| **OpenFace** | 128 | 92.92% | Very Fast | Edge devices |
| **DeepFace** | 4,096 | 97.35% | Slow | Research |
| **DeepID** | 160 | 97.45% | Fast | Mobile apps |
| **Dlib** | 128 | 99.38% | Fast | Lightweight |
| **SFace** | 128 | 99.50% | Fast | Balanced |
| **GhostFaceNet** | 512 | 99.50+ | Very Fast | Edge computing |
| **Buffalo_L** | 512 | 99.60+ | Fast | High performance |

### Detection Backends

| Backend | Speed | mAP | Use Case |
|---------|-------|-----|----------|
| **opencv** | ⚡⚡⚡⚡ | 85% | Quick detection |
| **ssd** | ⚡⚡⚡ | 90% | Balanced |
| **mtcnn** | ⚡⚡ | 95% | Multiple faces |
| **retinaface** | ⚡⚡ | 98% | Production (best) |
| **mediapipe** | ⚡⚡⚡⚡ | 93% | Real-time mobile |
| **dlib** | ⚡⚡⚡ | 92% | Landmarks |
| **yolo** | ⚡⚡⚡ | 96% | Modern detection |
| **yunet** | ⚡⚡⚡ | 94% | OpenCV optimized |
| **centerface** | ⚡⚡⚡ | 95% | Anchor-free |

---

## REST API

### Start Server

```bash
cd deepface/api/src
python app.py
# Server runs on http://localhost:5000
```

### API Endpoints

#### POST `/verify`
```bash
curl -X POST http://localhost:5000/verify \
-H "Content-Type: application/json" \
-d '{
  "img1_path": "path/to/img1.jpg",
  "img2_path": "path/to/img2.jpg",
  "model_name": "VGG-Face",
  "detector_backend": "retinaface"
}'
```

#### POST `/analyze`
```bash
curl -X POST http://localhost:5000/analyze \
-H "Content-Type: application/json" \
-d '{
  "img_path": "path/to/image.jpg",
  "actions": ["age", "gender", "emotion", "race"]
}'
```

#### POST `/find`
```bash
curl -X POST http://localhost:5000/find \
-H "Content-Type: application/json" \
-d '{
  "img_path": "path/to/target.jpg",
  "db_path": "face_database/",
  "model_name": "Facenet"
}'
```

#### POST `/represent`
```bash
curl -X POST http://localhost:5000/represent \
-H "Content-Type: application/json" \
-d '{
  "img_path": "path/to/face.jpg",
  "model_name": "Facenet"
}'
```

---

## Advanced Usage

### Database Integration

#### Weaviate (Vector Database)

```python
from deepface.modules.database import weaviate

# Initialize client
db = weaviate.WeaviateClient(
    url="http://localhost:8080",
    api_key=os.getenv("WEAVIATE_API_KEY")
)

# Store face embeddings
embedding = DeepFace.represent("person.jpg", model_name="Facenet")
db.store_embedding(
    vector=embedding,
    metadata={"name": "John Doe", "employee_id": "12345"}
)

# Search similar faces
results = db.search(query_vector=embedding, limit=10)
```

#### PostgreSQL

```python
from deepface.modules.database import postgres

db = postgres.PostgresClient(
    host="localhost",
    database="deepface_db",
    user="admin",
    password=os.getenv("DB_PASSWORD")
)

# Store face data
db.insert_face(
    name="John Doe",
    embedding=embedding.tolist(),
    metadata={"department": "Engineering"}
)

# Query faces
matches = db.find_similar(embedding, threshold=0.6, limit=5)
```

#### MongoDB

```python
from deepface.modules.database import mongo

db = mongo.MongoClient(
    uri="mongodb://localhost:27017",
    database_name="deepface"
)

# Insert face record
db.faces.insert_one({
    "name": "Jane Smith",
    "embedding": embedding.tolist(),
    "timestamp": datetime.now()
})
```

### Batch Processing

```python
import os
from deepface import DeepFace

# Process multiple images
image_folder = "path/to/images/"
results = []

for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(image_folder, filename)
        try:
            result = DeepFace.analyze(img_path, enforce_detection=False)
            results.append({
                'filename': filename,
                'age': result['age'],
                'gender': result['dominant_gender'],
                'emotion': result['dominant_emotion']
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('analysis_results.csv', index=False)
```

### Custom Thresholds

```python
from deepface.config import threshold

# Get default thresholds
thresholds = threshold.get_threshold("VGG-Face", "cosine")
print(f"Default threshold: {thresholds}")

# Verify with custom threshold
result = DeepFace.verify("img1.jpg", "img2.jpg", model_name="Facenet")
custom_threshold = 0.5
is_match = result['distance'] < custom_threshold
```

### Anti-Spoofing

```python
from deepface import DeepFace

# Check if face is real or spoofed
result = DeepFace.extract_faces(
    img_path="face.jpg",
    detector_backend="retinaface",
    anti_spoofing=True  # Enable liveness detection
)

for face in result:
    if face['is_real']:
        print("Live face detected")
    else:
        print("Spoofing attempt detected!")
```

### Homomorphic Encryption

```python
from deepface.modules import encryption

# Encrypt embeddings for privacy
embedding = DeepFace.represent("face.jpg", model_name="Facenet")
encrypted = encryption.encrypt(embedding)

# Compare encrypted embeddings
encrypted2 = encryption.encrypt(DeepFace.represent("face2.jpg"))
distance = encryption.calculate_distance(encrypted, encrypted2)
```

---

## Configuration Examples

### Face Database Setup

```
face_database/
├── person1/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── person2/
│   ├── img1.jpg
│   └── img2.jpg
└── person3/
    └── face.jpg
```

Each subdirectory name becomes the identity label.

### Environment Variables

Create `.env` file:

```bash
# Database
WEAVIATE_API_KEY=your_api_key
POSTGRES_PASSWORD=your_password
MONGO_URI=mongodb://localhost:27017

# API
FLASK_ENV=production
FLASK_PORT=5000

# Model paths
MODEL_WEIGHTS_PATH=/path/to/weights
```

---

## Installation

### Standard Installation

```bash
pip install -r requirements.txt
```

### From Source

```bash
git clone https://github.com/abhinavkarra/DeepFace-Master.git
cd DeepFace-Master
pip install -e .
```

### Dependencies

```
tensorflow>=1.9.0
keras>=2.2.0
opencv-python>=4.5.5
numpy>=1.14.0
pandas>=0.23.4
requests>=2.27.1
Pillow>=5.2.0
flask>=1.1.2
mtcnn>=0.1.0
retina-face>=0.0.14
```

---

## Performance Tips

1. **Use GPU**: Install `tensorflow-gpu` for 10x speedup
2. **Choose Right Model**: Facenet/QMAG for speed, VGG-Face for accuracy
3. **Detector Selection**: Use `opencv` for speed, `retinaface` for accuracy
4. **Batch Processing**: Process multiple frames before analysis
5. **Frame Skipping**: In video, process every Nth frame
6. **Disable Detection**: Use `enforce_detection=False` when face is guaranteed

---

## Troubleshooting

### Common Issues

**No face detected:**
```python
# Try different detector or disable enforcement
result = DeepFace.verify("img1.jpg", "img2.jpg", 
                         detector_backend="opencv",
                         enforce_detection=False)
```

**Memory issues:**
```python
# Use lightweight model
DeepFace.verify("img1.jpg", "img2.jpg", model_name="OpenFace")
```

**Slow performance:**
```python
# Use faster detector
DeepFace.analyze("img.jpg", detector_backend="opencv")
```

---

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and design
- [DETAILED_PIPELINE.md](DETAILED_PIPELINE.md) - Complete processing pipeline
- [ACCURACY_ANALYSIS.md](ACCURACY_ANALYSIS.md) - Benchmark results

---

## License

MIT License - see [LICENSE](LICENSE) file

---

## Contributing

Pull requests welcome! See issues for feature requests.

```bash
git checkout -b feature/amazing-feature
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

---

<div align="center">

**[GitHub](https://github.com/abhinavkarra/DeepFace-Master)** • **[Issues](https://github.com/abhinavkarra/DeepFace-Master/issues)** • **[Discussions](https://github.com/abhinavkarra/DeepFace-Master/discussions)**

Made for developers who need production-ready face recognition

</div>
