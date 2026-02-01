from classifier import Classifier


def main():
    print("starting classification")
    Classifier.New("resources/blah_basketball.mp4", True)
    print("finished classification")


if __name__ == "__main__":
    main()
