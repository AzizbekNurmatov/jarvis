"""Download the official MediaPipe face landmarker model on first launch."""
from __future__ import annotations

from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def ensure_model(path: Path) -> Path:
    if path.is_file() and path.stat().st_size > 100_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print("First launch: downloading the official MediaPipe Face Landmarker model...")
    temp = path.with_suffix(".download")
    try:
        with urlopen(MODEL_URL, timeout=45) as response, temp.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if temp.stat().st_size < 100_000:
            raise RuntimeError("Downloaded model is unexpectedly small.")
        temp.replace(path)
    except (OSError, URLError, RuntimeError) as exc:
        temp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Couldn't download model: {exc}\n"
            f"Download it manually from:\n{MODEL_URL}\n"
            f"and save it to: {path}"
        ) from exc
    print(f"Model saved to {path}")
    return path
