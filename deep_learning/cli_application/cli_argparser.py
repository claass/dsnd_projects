import argparse


def train_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to input data directory')
    parser.add_argument(
        '--save_dir',
        type=str,
        required=False,
        help='Path for saving model checkpoint')
    parser.add_argument(
        '--arch',
        type=str,
        default='resnet18',
        help='Architecture for the pretrained base [resnet18, resnet152]')
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='Learning rate [default: 0.001]',
        default=0.001)
    parser.add_argument(
        '--hidden_units',
        type=int,
        help='Number of hidden units [default: 500]',
        default=500)
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of trainig epochs [default: 3]',
        default=3)
    parser.add_argument(
        '--gpu',
        action="store_true",
        help='Activates gpu usage',
        default=False,)

    return parser.parse_args()


def predict_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'im_path',
        type=str,
        help='Path to image')
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint')
    parser.add_argument(
        '--top_k',
        type=int,
        default=1,
        help='Returns N most likely labels [default: 1]')
    parser.add_argument(
        '--category_names',
        type=str,
        help='Path to catgory_to_name mapping in JSON format',
        required=False)
    parser.add_argument(
        '--gpu',
        action="store_true",
        help='Activates gpu usage',
        default=False)

    return parser.parse_args()
