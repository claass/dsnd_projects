
import transfer_learning as tl
from cli_argparser import predict_parser


if __name__ == '__main__':

    # Get user inputs from the cli
    args = predict_parser()

    model, criterion, optimizer, version = tl.load_from_checkpoint(
        args.checkpoint)

    proba, output_label = tl.predict(
        args.im_path,
        model,
        topk=args.top_k,
        class_names_json=args.category_names,
        use_gpu=args.gpu)

    for result in zip(output_label, proba):
        print(result)
