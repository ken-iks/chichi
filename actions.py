from dataclasses import dataclass
from typing import Literal


@dataclass
class Pass:
    source: int
    dest: int


@dataclass
class Shoot:
    source: int
    pts: Literal[0, 1, 2, 3]


@dataclass
class PlayerAction:
    frame: int
    action: Pass | Shoot
