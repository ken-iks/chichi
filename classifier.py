from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

from embeddings import EmbeddingModel
from teams import Player, Team, load_ball_positions, load_teams, save_ball_positions, save_teams

PERSISTANCE_PATH = "output/teams.json"
BALL_PERSISTANCE_PATH = "output/ball_positions.json"


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
        self._embed_model: EmbeddingModel | None = None

    @staticmethod
    def New(fp: str, persist: bool, yolo=None, embed_model=None) -> "Classifier":
        if not Path(fp).exists():
            raise ValueError("Invalid file path")

        model = yolo if yolo is not None else YOLO("yolo26n.pt")
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

    def _get_embedding(self, frame: NDArray[np.uint8], box: Boxes) -> np.ndarray:
        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
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

        if frame_idx == 0:
            # instantiate teams on first frame
            crops = []
            track_ids_list = []
            for box, track_id in people:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                cropped = result.orig_img[y1:y2, x1:x2]
                crops.append(cropped)
                track_ids_list.append(track_id)

            embeddings_array = self._embed_model.get_embeddings_batch(crops)
            for i, track_id in enumerate(track_ids_list):
                self._embeddings[track_id] = embeddings_array[i]

            kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
            labels = kmeans.fit_predict(embeddings_array)

            cluster_sizes = [np.sum(labels == i) for i in range(3)]
            sorted_clusters = sorted(range(3), key=lambda i: cluster_sizes[i], reverse=True)
            team1_label, team2_label = sorted_clusters[0], sorted_clusters[1]

            self._team_centroids = (
                kmeans.cluster_centers_[team1_label],
                kmeans.cluster_centers_[team2_label],
            )

            team1_crops = [crops[i] for i, label in enumerate(labels) if label == team1_label]
            team2_crops = [crops[i] for i, label in enumerate(labels) if label == team2_label]
            color1 = self._embed_model.get_color_from_crops(team1_crops)
            color2 = self._embed_model.get_color_from_crops(team2_crops)
            self.teams = (Team(color1), Team(color2))

            label_map = {team1_label: 0, team2_label: 1}
            labels = np.array([label_map.get(l, -1) for l in labels])

            for idx, (box, track_id) in enumerate(people):
                label = labels[idx]
                if label == -1:
                    continue
                player = Player(track_id)
                coords = tuple(int(x) for x in box.xyxy[0].cpu().numpy())
                player.add_position(frame_idx, coords)
                self.teams[label].players[track_id] = player
        else:
            known_ids = self._get_all_player_ids()
            for box, track_id in people:
                coords = tuple(int(x) for x in box.xyxy[0].cpu().numpy())
                if track_id not in known_ids:
                    emb = self._get_embedding(result.orig_img, box)
                    self._embeddings[track_id] = emb
                    team, _ = self._closest_team(emb)
                    player = Player(track_id)
                    player.add_position(frame_idx, coords)
                    team.players[track_id] = player
                else:
                    for team in self.teams:
                        if track_id in team.players:
                            team.players[track_id].add_position(frame_idx, coords)
                            break
