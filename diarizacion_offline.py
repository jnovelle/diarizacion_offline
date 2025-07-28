from pyannote.audio.pipelines import SpeakerDiarization
from pathlib import Path

# Ruta absoluta al pipeline local
pipeline_path = Path(r"C:\Users\JoseManuel\_AI\diarizacion_offline\modelos\diarization")

# Cargar pipeline local
pipeline = SpeakerDiarization.from_pretrained(pipeline_path)

# Ruta al audio
audio_path = Path(r"C:\Users\JoseManuel\_AI\diarizacion_offline\audio.mp3")

# Procesar diarización
diarization = pipeline(audio_path)

# Mostrar resultados
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f} - {turn.end:.1f} : {speaker}")
