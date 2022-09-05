import argparse


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="voc/2007")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="/Users/wonhyung64/data")
    parser.add_argument("--img-size", nargs="+", type=int, default=[512, 512])
    parser.add_argument(
        "--variances", nargs="+", type=float, default=[0.1, 0.1, 0.2, 0.2]
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weights-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--prob_init", type=float, default=0.01)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    return args
