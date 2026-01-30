def load_labels(path="imagenet_classes.txt"):
    with open(path) as f:
        return [line.strip() for line in f.readlines()]
