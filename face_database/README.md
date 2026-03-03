# Face Database Directory

This directory is used for storing face images for recognition and verification tasks.

## Structure

- Place images in subdirectories named after individuals (e.g., `Person1/`, `Person2/`)
- Each subdirectory should contain one or more images of that person
- Supported formats: JPG, JPEG, PNG

## Usage

When using DeepFace's `find()` function, specify this directory as the database path:

```python
from deepface import DeepFace

result = DeepFace.find(
    img_path="target_image.jpg",
    db_path="face_database"
)
```

## Note

- Cache files (*.pkl) are automatically generated and should not be committed to version control
- Keep personal/sensitive images out of public repositories
