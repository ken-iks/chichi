from ultralytics import YOLO

from classifier import BALL_PERSISTANCE_PATH, PERSISTANCE_PATH, Classifier
from merge import merge_fragmented_tracks
from teams import save_ball_positions, save_teams


def main():
    model = YOLO("yolo26n-seg.pt")
    print("loaded")


def process_video(fp: str):
    cl = Classifier.New(fp, persist=False)

    print(
        f"Before merge: Team 1 has {len(cl.teams[0].players)}, Team 2 has {len(cl.teams[1].players)}"
    )
    merge_fragmented_tracks(cl.teams, cl._embeddings)
    print(
        f"After merge: Team 1 has {len(cl.teams[0].players)}, Team 2 has {len(cl.teams[1].players)}"
    )

    save_teams(cl.teams, PERSISTANCE_PATH)
    save_ball_positions(cl.ball_positions, BALL_PERSISTANCE_PATH)

    return cl


if __name__ == "__main__":
    main()
