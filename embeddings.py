import json

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, CLIPModel, CLIPProcessor, Qwen2VLForConditionalGeneration


class EmbeddingModel:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.float16,
        ).to(self.device)
        self.vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    def _extract_clip_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model.get_image_features(pixel_values=pixel_values)
        if isinstance(outputs, torch.Tensor):
            return outputs
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            return outputs.image_embeds
        if hasattr(outputs, "pooler_output"):
            pooled = outputs.pooler_output
            if pooled.shape[-1] == self.model.config.projection_dim:
                return pooled
            return self.model.visual_projection(pooled)
        if isinstance(outputs, tuple):
            return outputs[0]
        raise RuntimeError(f"Unexpected CLIP output type: {type(outputs)}")

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            features = self._extract_clip_features(inputs["pixel_values"])
        return features.detach().cpu().numpy().flatten()

    def get_embeddings_batch(self, images: list[np.ndarray]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            features = self._extract_clip_features(inputs["pixel_values"])
        return features.detach().cpu().numpy()

    def get_team_colors_from_scene(self, frame: np.ndarray) -> tuple[str, str]:
        image = Image.fromarray(frame)
        content = [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": (
                    "This is a sports match scene. Identify the two main team jersey colors visible. "
                    "Ignore referees and spectators. Focus only on the outfield players. "
                    "Answer in strict JSON only, no extra text, with this format:\n"
                    '{"team_1_color": "<main jersey color name>", "team_2_color": "<main jersey color name>"}'
                ),
            },
        ]

        messages = [{"role": "user", "content": content}]
        image_inputs, video_inputs = process_vision_info(messages)
        text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self.vlm.generate(**inputs, max_new_tokens=64)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return self._parse_team_colors_json(output_text)

    def classify_player_by_color(self, crop: np.ndarray, team_colors: tuple[str, str]) -> int:
        image = Image.fromarray(crop)
        content = [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": (
                    f"This is a cropped image of a sports player. "
                    f"The two team colors are: team 1 = {team_colors[0]}, team 2 = {team_colors[1]}. "
                    f"Which team does this player belong to based on their jersey color? "
                    f"If they appear to be a referee or goalkeeper, answer -1. "
                    f"Answer with only a single number: 1, 2, or -1"
                ),
            },
        ]

        messages = [{"role": "user", "content": content}]
        image_inputs, video_inputs = process_vision_info(messages)
        text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self.vlm.generate(**inputs, max_new_tokens=8)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        try:
            val = int(output_text.strip())
            if val == 1:
                return 0
            elif val == 2:
                return 1
            return -1
        except ValueError:
            return -1

    def _parse_team_colors_json(self, response: str) -> tuple[str, str]:
        try:
            obj = json.loads(response)
            c1 = obj.get("team_1_color", "unknown").strip().lower()
            c2 = obj.get("team_2_color", "unknown").strip().lower()
            return (c1, c2)
        except json.JSONDecodeError:
            return ("unknown", "unknown")
