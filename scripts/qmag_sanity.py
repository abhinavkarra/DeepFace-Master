import sys
print("Python:", sys.version)

try:
    from deepface.models.facial_recognition.qmag import qmagClient
except Exception as e:
    print("Import failed:", e)
    raise

print("Instantiating qmagClient...")
client = qmagClient()
print("Model:", client.model_name)
print("Input shape:", client.input_shape)
print("Output shape:", client.output_shape)

layers = [layer.name for layer in client.model.layers]
print("Layers count:", len(layers))
print("First layer:", layers[0] if layers else "-" )
print("Last layer:", layers[-1] if layers else "-" )
