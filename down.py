from huggingface_hub import snapshot_download

# Descargar diarization
snapshot_download("pyannote/speaker-diarization-3.1", local_dir="modelos/diarization")

# Descargar segmentation
snapshot_download("pyannote/segmentation-3.0", local_dir="modelos/segmentation")

# Descargar embedding
snapshot_download("pyannote/embedding", local_dir="modelos/embedding")

print("Modelos descargados correctamente.")
