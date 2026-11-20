"""
Scarica i modelli necessari per il riconoscimento delle voci nella cartella models/.
Il modello Whisper viene scaricato automaticamente al primo utilizzo.

Uso:
    python scarica_modelli.py
"""
import tarfile
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent / "models"
REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTAZIONE = f"{REL}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
IMPRONTE = f"{REL}/speaker-recongition-models/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"


def scarica(url: str, dest: Path):
    if dest.exists():
        print(f"Già presente: {dest.name}")
        return
    print(f"Scarico {dest.name}...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def main():
    (BASE / "emb").mkdir(parents=True, exist_ok=True)
    seg_dir = BASE / "sherpa-onnx-pyannote-segmentation-3-0"
    if not (seg_dir / "model.onnx").exists():
        archivio = BASE / "segmentazione.tar.bz2"
        scarica(SEGMENTAZIONE, archivio)
        with tarfile.open(archivio) as tar:
            tar.extractall(BASE, filter="data")
        archivio.unlink()
    else:
        print("Già presente: modello di segmentazione")
    scarica(IMPRONTE, BASE / "emb" / IMPRONTE.rsplit("/", 1)[1])
    print("Modelli pronti.")


if __name__ == "__main__":
    main()
