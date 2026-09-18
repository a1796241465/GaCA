from ablations.models import MeanConcat
from gaca import train as base


ORIGINAL_BUILD_MODEL = base.build_model


def build_model(name, num_classes, args):
    if name == "mean_concat":
        return MeanConcat(num_classes, args.dropout)
    return ORIGINAL_BUILD_MODEL(name, num_classes, args)


def main():
    base.build_model = build_model
    base.main()


if __name__ == "__main__":
    main()
