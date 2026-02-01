import json
import tempfile
from pathlib import Path

import modal

app = modal.App("chichi")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics",
        "numpy",
        "scikit-learn",
        "torch",
        "transformers==4.40.0",
        "lap",
    )
    .add_local_file("yolo26n.pt", "/root/yolo26n.pt")
    .add_local_dir(".", "/root/app")
)

with image.imports():
    import sys

    sys.path.insert(0, "/root/app")
    from ultralytics import YOLO

    from classifier import Classifier
    from embeddings import EmbeddingModel
    from merge import merge_fragmented_tracks


@app.cls(gpu="A100", image=image, timeout=3600)
class Worker:
    @modal.enter()
    def setup(self):
        self.yolo = YOLO("/root/yolo26n.pt")
        self.embed_model = EmbeddingModel()

    @modal.method()
    def process_video(self, video_bytes: bytes) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            video_path = f.name

        cl = Classifier.New(video_path, persist=False, yolo=self.yolo, embed_model=self.embed_model)
        merge_fragmented_tracks(cl.teams, cl._embeddings)

        return {
            "teams": [t.to_dict() for t in cl.teams],
            "ball_positions": {str(k): list(v) for k, v in cl.ball_positions.items()},
        }


@app.local_entrypoint()
def main(video_path: str = "resources/blah_basketball.mp4", output_dir: str = "output"):
    video_bytes = Path(video_path).read_bytes()
    result = Worker().process_video.remote(video_bytes)

    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    (out / "teams.json").write_text(json.dumps(result["teams"], indent=2))
    (out / "ball_positions.json").write_text(json.dumps(result["ball_positions"], indent=2))
