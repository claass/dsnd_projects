
import transfer_learning as tl
from torch import optim
from torch import nn
from cli_argparser import train_parser

if __name__ == '__main__':

    # Get user inputs from the cli
    args = train_parser()

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
    tl.train_model(
        model, optimizer, criterion, data, num_epochs=args.epochs,
        calc_validation=True, device=device)

    # IF a checkpoint_path was provided, save to checkpoint
    if args.save_dir:
        tl.save_checkpoint(model, optimizer, criterion, data, args.arch, 1)
