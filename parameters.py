import argparse


def return_args():
    parser = argparse.ArgumentParser(description="Parameters in project")

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--dataset_loc", type=str, default="MovieChat//Data")
    parser.add_argument("--tokenizer_name", type=str, default="nltk")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--valid_ratio", type=float, default=0.2)

    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)

    args = parser.parse_args()

    return args
