
import argparse
import transfer_learning as tl
from torch import optim
from torch import nn


def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'data_dir',
        type=str,
        help='What is the path to the data directory?'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        required=False,
        help='Where should the model be saved to?'
    )
    parser.add_argument(
        '--arch',
        type=str,
        help='What base architecture should be used? [resnet18, resnet152]',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='What learning rate should be used? [default: 0.001]',
        default=0.001
    )
    parser.add_argument(
        '--hidden_units',
        type=int,
        help='How many hidden units should be used? [default: 500]',
        default=500
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='For how many epochs should be trained? [default: 3]',
        default=3
    )
    parser.add_argument(
        '--gpu',
        action="store_true",
        help='Activates gpu usage',
        default=False,
    )
    return parser.parse_args()

    # sys.stdout.write(str(calc(args)))


if __name__ == '__main__':

    # Get user inputs from the cli
    args = parse_cli()

    # Load the data from the data dict
    data = tl.load_data_from_dir(args.data_dir)

    # Load the specified model
    model = tl.load_pretrained_model(args.arch)

    # Replace the classifier as per users specification
    tl.replace_classifier(model, hidden_units=args.hidden_units)

    # Start training the model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    device = 'cuda' if args.gpu else 'cpu'
    tl.train_model(model, optimizer, criterion, data, num_epochs=args.epochs,
                   calc_validation=True, device=device)

    # IF a checkpoint_path was provided, save to checkpoint
    if args.save_dir:
        tl.save_checkpoint(model, optimizer, criterion, data, args.arch, 1)
