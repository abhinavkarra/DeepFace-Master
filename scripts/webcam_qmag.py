"""
Run DeepFace webcam stream using the qmag recognition model.
"""
import os
import sys

# Ensure local package is importable when running from scripts/
try:
    from deepface import DeepFace
except ModuleNotFoundError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    from deepface import DeepFace

# Database path for recognition
DB_PATH = os.path.join(os.path.dirname(__file__), "../face_database")

print("Using database:", os.path.abspath(DB_PATH))
print("Model: qmag | Detector: retinaface | Metric: cosine")
print("Press 'q' to quit")

DeepFace.stream(
    db_path=DB_PATH,
    model_name="qmag",
    detector_backend="retinaface",
    distance_metric="cosine",
    enable_face_analysis=True,
    source=0,
    time_threshold=5,
    frame_threshold=5,
    anti_spoofing=False,
    output_path=None,
    debug=False,
)
