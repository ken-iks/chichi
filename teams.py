import json
from dataclasses import asdict
from pathlib import Path

from actions import Pass, PlayerAction, Shoot


class Player:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.positions: dict[int, tuple[int, int, int, int]] = {}
        self.actions: list[PlayerAction] = []

    def add_position(self, frame_idx: int, position: tuple[int, int, int, int]):
        self.positions[frame_idx] = position

    def get_position(self, frame_idx: int) -> tuple[int, int, int, int] | None:
        return self.positions.get(frame_idx)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "positions": {str(k): list(v) for k, v in self.positions.items()},
            "actions": [asdict(a) for a in self.actions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Player":
        player = cls(data["track_id"])
        player.positions = {int(k): tuple(v) for k, v in data["positions"].items()}
        for a in data["actions"]:
            action_data = a["action"]
            if "dest" in action_data:
                action = Pass(**action_data)
            else:
                action = Shoot(**action_data)
            player.actions.append(PlayerAction(frame=a["frame"], action=action))
        return player


class Team:
    def __init__(self, color: tuple[int, int, int]):
        self.team_color = color
        self.players: dict[int, Player] = {}

    def to_dict(self) -> dict:
        return {
            "team_color": list(self.team_color),
            "players": {str(k): v.to_dict() for k, v in self.players.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Team":
        team = cls(tuple(data["team_color"]))
        team.players = {int(k): Player.from_dict(v) for k, v in data["players"].items()}
        return team


def save_teams(teams: tuple["Team", "Team"], path: str):
    data = [t.to_dict() for t in teams]
    Path(path).write_text(json.dumps(data, indent=2))


def load_teams(path: str) -> tuple["Team", "Team"]:
    data = json.loads(Path(path).read_text())
    return (Team.from_dict(data[0]), Team.from_dict(data[1]))
