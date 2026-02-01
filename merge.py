from collections import defaultdict

import numpy as np

from teams import Team


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def tracks_overlap(p1, p2) -> bool:
    return bool(set(p1.positions.keys()) & set(p2.positions.keys()))


def spatially_plausible(p1, p2, max_distance=200) -> bool:
    frames1 = sorted(p1.positions.keys())
    frames2 = sorted(p2.positions.keys())
    if not frames1 or not frames2:
        return False

    if frames1[-1] < frames2[0]:
        last_pos = p1.positions[frames1[-1]]
        first_pos = p2.positions[frames2[0]]
    elif frames2[-1] < frames1[0]:
        last_pos = p2.positions[frames2[-1]]
        first_pos = p1.positions[frames1[0]]
    else:
        return False

    cx1, cy1 = (last_pos[0] + last_pos[2]) / 2, (last_pos[1] + last_pos[3]) / 2
    cx2, cy2 = (first_pos[0] + first_pos[2]) / 2, (first_pos[1] + first_pos[3]) / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5 < max_distance


def find_merge_candidates(team: Team, embeddings: dict[int, np.ndarray], similarity_threshold=0.88):
    pairs = []
    track_ids = list(team.players.keys())
    for i, id1 in enumerate(track_ids):
        for id2 in track_ids[i + 1 :]:
            if id1 not in embeddings or id2 not in embeddings:
                continue
            p1, p2 = team.players[id1], team.players[id2]
            if tracks_overlap(p1, p2):
                continue
            if cosine_similarity(embeddings[id1], embeddings[id2]) > similarity_threshold:
                if spatially_plausible(p1, p2):
                    pairs.append((id1, id2))
    return pairs


def merge_fragmented_tracks(
    teams: tuple[Team, Team],
    embeddings: dict[int, np.ndarray],
    similarity_threshold=0.88,
):
    for team in teams:
        pairs = find_merge_candidates(team, embeddings, similarity_threshold)

        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            parent[find(b)] = find(a)

        for id1, id2 in pairs:
            union(id1, id2)

        groups = defaultdict(list)
        for track_id in parent:
            groups[find(track_id)].append(track_id)

        for root, members in groups.items():
            for member in members:
                if member != root and member in team.players:
                    p_root = team.players[root]
                    p_member = team.players[member]
                    p_root.positions.update(p_member.positions)
                    p_root.actions.extend(p_member.actions)
                    del team.players[member]
