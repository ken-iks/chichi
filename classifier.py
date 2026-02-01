from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

from embeddings import EmbeddingModel
from teams import Player, Team, load_ball_positions, load_teams, save_ball_positions, save_teams

PERSISTANCE_PATH = "output/teams.json"
BALL_PERSISTANCE_PATH = "output/ball_positions.json"


def apply_mask_to_crop(frame: np.ndarray, box_coords: tuple, mask: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box_coords
    cropped = frame[y1:y2, x1:x2].copy()
    mask_cropped = mask[y1:y2, x1:x2]
    cropped[mask_cropped == 0] = 0
    return cropped


class Classifier:
    def __init__(
        self,
        _teams: tuple[Team, Team] = None,
        _ball_positions: dict[int, tuple[int, int, int, int]] = None,
    ):
        self.teams = _teams
        self.ball_positions = _ball_positions if _ball_positions else {}
        self._embeddings: dict[int, np.ndarray] = {}
        self._team_centroids: tuple[np.ndarray, np.ndarray] | None = None
        self._team_colors: tuple[str, str] | None = None
        self._embed_model: EmbeddingModel | None = None

    @staticmethod
    def New(fp: str, persist: bool, yolo=None, embed_model=None) -> "Classifier":
        if not Path(fp).exists():
            raise ValueError("Invalid file path")

        model = yolo if yolo is not None else YOLO("yolo26n-seg.pt")

        results = model.track(
            source=fp,
            tracker="bytetrack.yaml",
            classes=[0, 32],  # 0=people, 32=ball
            stream=True,
        )
        cl = Classifier()
        cl._embed_model = embed_model if embed_model is not None else EmbeddingModel()

        for i, res in enumerate(results):
            if res.boxes is not None:
                cl._assign_teams(res, i)

        if persist:
            if cl.teams is not None and len(cl.ball_positions) > 0:
                save_teams(cl.teams, PERSISTANCE_PATH)
                save_ball_positions(cl.ball_positions, BALL_PERSISTANCE_PATH)
            else:
                raise ValueError("Error persisting", "teams", cl.teams, "balls", cl.ball_positions)

        return cl

    @staticmethod
    def From_saved() -> "Classifier":
        return Classifier(load_teams(PERSISTANCE_PATH), load_ball_positions(BALL_PERSISTANCE_PATH))

    def _get_embedding(
        self, frame: NDArray[np.uint8], box: Boxes, mask: np.ndarray = None
    ) -> np.ndarray:
        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        if mask is not None:
            cropped = apply_mask_to_crop(frame, (x1, y1, x2, y2), mask)
        else:
            cropped = frame[y1:y2, x1:x2]
        return self._embed_model.get_embedding(cropped)

    def _closest_team(self, embedding: np.ndarray) -> tuple[Team, int]:
        dist1 = np.linalg.norm(embedding - self._team_centroids[0])
        dist2 = np.linalg.norm(embedding - self._team_centroids[1])
        if dist1 < dist2:
            return self.teams[0], 0
        return self.teams[1], 1

    def _get_all_player_ids(self) -> set[int]:
        ids = set()
        for team in self.teams:
            ids.update(team.players.keys())
        return ids

    def _get_masks_dict(self, result: Results) -> dict[int, np.ndarray]:
        masks_dict = {}
        if result.masks is None:
            return masks_dict
        h, w = result.orig_img.shape[:2]
        for i, box in enumerate(result.boxes):
            if box.id is None:
                continue
            track_id = int(box.id.item())
            mask_tensor = result.masks.data[i]
            mask_resized = np.array(
                Image.fromarray(mask_tensor.cpu().numpy()).resize((w, h), Image.NEAREST)
            )
            masks_dict[track_id] = mask_resized
        return masks_dict

    def _assign_teams(self, result: Results, frame_idx: int):
        people = [
            (box, int(box.id.item()))
            for box in result.boxes
            if box.cls.item() == 0 and box.id is not None
        ]

        balls = [box for box in result.boxes if box.cls.item() == 32 and box.id is not None]
        if balls:
            box = balls[0]
            coords = tuple(int(x) for x in box.xyxy[0].cpu().numpy())
            self.ball_positions[frame_idx] = coords

        masks_dict = self._get_masks_dict(result)

        if frame_idx == 0:
            color1, color2 = self._embed_model.get_team_colors_from_scene(result.orig_img)
            self._team_colors = (color1, color2)
            self.teams = (Team(color1), Team(color2))

            crops = []
            track_ids_list = []
            for box, track_id in people:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                mask = masks_dict.get(track_id)
                if mask is not None:
                    cropped = apply_mask_to_crop(result.orig_img, (x1, y1, x2, y2), mask)
                else:
                    cropped = result.orig_img[y1:y2, x1:x2]
                crops.append(cropped)
                track_ids_list.append(track_id)

            embeddings_array = self._embed_model.get_embeddings_batch(crops)
            for i, track_id in enumerate(track_ids_list):
                self._embeddings[track_id] = embeddings_array[i]

            team1_embeddings = []
            team2_embeddings = []

            for idx, (box, track_id) in enumerate(people):
                coords = tuple(int(x) for x in box.xyxy[0].cpu().numpy())
                label = self._embed_model.classify_player_by_color(crops[idx], self._team_colors)
                if label == -1:
                    continue
                player = Player(track_id)
                player.add_position(frame_idx, coords)
                self.teams[label].players[track_id] = player

                if label == 0:
                    team1_embeddings.append(embeddings_array[idx])
                else:
                    team2_embeddings.append(embeddings_array[idx])

            if team1_embeddings and team2_embeddings:
                self._team_centroids = (
                    np.mean(team1_embeddings, axis=0),
                    np.mean(team2_embeddings, axis=0),
                )
        else:
            known_ids = self._get_all_player_ids()
            for box, track_id in people:
                coords = tuple(int(x) for x in box.xyxy[0].cpu().numpy())
                if track_id not in known_ids:
                    mask = masks_dict.get(track_id)
                    emb = self._get_embedding(result.orig_img, box, mask)
                    self._embeddings[track_id] = emb

                    if self._team_centroids is not None:
                        team, _ = self._closest_team(emb)
                    else:
                        x1, y1, x2, y2 = coords
                        if mask is not None:
                            crop = apply_mask_to_crop(result.orig_img, (x1, y1, x2, y2), mask)
                        else:
                            crop = result.orig_img[y1:y2, x1:x2]
                        label = self._embed_model.classify_player_by_color(crop, self._team_colors)
                        if label == -1:
                            continue
                        team = self.teams[label]

                    player = Player(track_id)
                    player.add_position(frame_idx, coords)
                    team.players[track_id] = player
                else:
                    for team in self.teams:
                        if track_id in team.players:
                            team.players[track_id].add_position(frame_idx, coords)
                            break
