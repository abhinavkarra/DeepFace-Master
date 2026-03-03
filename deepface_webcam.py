 #!/usr/bin/env python3
"""
DeepFace Live - Full face recognition and analysis using webcam
"""
import os
from deepface import DeepFace

# Create a database directory if it doesn't exist
db_path = os.path.join(os.path.dirname(__file__), "face_database")
if not os.path.exists(db_path):
    os.makedirs(db_path)
    print(f"Created database directory: {db_path}")
    print(f"Add face images to this directory for recognition.")
    print(f"You can organize by person: face_database/PersonName/image.jpg")
    print()

print("=" * 60)
print("DeepFace Live - Real-time Face Recognition & Analysis")
print("=" * 60)
print()
print("Features:")
print("  • Face Detection")
print("  • Face Recognition (matches against database)")
print("  • Age & Gender Detection")
print("  • Emotion Recognition")
print("  • Race/Ethnicity Detection")
print()
print("Press 'q' to quit")
print("=" * 60)
print()

# Run the stream function
try:
    DeepFace.stream(db_path=db_path)
except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    print("\nNote: Make sure your webcam is not being used by another application")
