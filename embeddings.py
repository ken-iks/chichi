import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


class EmbeddingModel:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(pixel_values=pixel_values)
        return features.detach().cpu().numpy().flatten()

    def get_embeddings_batch(self, images: list[np.ndarray]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(pixel_values=pixel_values)
        return features.detach().cpu().numpy()

    def get_color_from_crops(self, crops: list[np.ndarray]) -> str:
        color_options = [
            "person wearing white jersey",
            "person wearing green jersey",
            "person wearing red jersey",
            "person wearing blue jersey",
            "person wearing black jersey",
            "person wearing yellow jersey",
            "person wearing orange jersey",
            "person wearing purple jersey",
        ]

        torso_crops = []
        for crop in crops:
            h = crop.shape[0]
            top = int(h * 0.1)
            bottom = int(h * 0.6)
            torso = crop[top:bottom, :]
            if torso.size > 0:
                torso_crops.append(torso)

        if not torso_crops:
            return "unknown"

        inputs = self.processor(
            text=color_options, images=torso_crops, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)

        avg_probs = probs.mean(dim=0)
        best_idx = avg_probs.argmax().item()
        return color_options[best_idx].replace("person wearing ", "").replace(" jersey", "")
