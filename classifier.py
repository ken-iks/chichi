from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

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

    @staticmethod
    def New(fp: str, persist: bool) -> "Classifier":
        if not Path(fp).exists():
            raise ValueError("Invalid file path")

        model = YOLO("yolo26n.pt")
        results = model.track(
            source=fp,
            tracker="bytetrack.yaml",
            classes=[0, 32],  # 0=people, 32=ball
            stream=True,
        )
        cl = Classifier()
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

    @staticmethod
    def _get_dominant_color(frame: NDArray[np.uint8], box: Boxes) -> NDArray[np.int_]:
        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        cropped = frame[y1:y2, x1:x2]  # numpy is row major so reshaped
        # flatten image into a list of pixels
        pixels = cropped.reshape(-1, 3).astype(np.float32)
        kmeans = KMeans(n_clusters=1, n_init=10)
        kmeans.fit(pixels)
        return kmeans.cluster_centers_[0].astype(np.int_)

    def _closest_team(self, color: NDArray[np.int_]) -> Team:
        dist1 = np.linalg.norm(color - np.array(self.teams[0].team_color))
        dist2 = np.linalg.norm(color - np.array(self.teams[1].team_color))
        return self.teams[0] if dist1 < dist2 else self.teams[1]

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
            colors = np.array(
                [self._get_dominant_color(result.orig_img, box) for box, _ in people]
            ).reshape(-1, 3)
            kmeans = KMeans(n_clusters=2, n_init=10)
            labels = kmeans.fit_predict(colors)
            team1_color = tuple(int(x) for x in kmeans.cluster_centers_[0])
            team2_color = tuple(int(x) for x in kmeans.cluster_centers_[1])

            self.teams = (Team(team1_color), Team(team2_color))

            for (box, track_id), label in zip(people, labels):
                player = Player(track_id)
                coords = tuple(int(x) for x in box.xyxy[0].cpu().numpy())
                player.add_position(frame_idx, coords)
                self.teams[label].players[track_id] = player
        else:
            known_ids = self._get_all_player_ids()
            for box, track_id in people:
                coords = tuple(int(x) for x in box.xyxy[0].cpu().numpy())
                if track_id not in known_ids:
                    color = self._get_dominant_color(result.orig_img, box)
                    team = self._closest_team(color)
                    player = Player(track_id)
                    player.add_position(frame_idx, coords)
                    team.players[track_id] = player
                else:
                    for team in self.teams:
                        if track_id in team.players:
                            team.players[track_id].add_position(frame_idx, coords)
                            break
