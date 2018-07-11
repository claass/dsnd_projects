

# Utility imports
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook, tqdm
from PIL import Image
from collections import OrderedDict

# DL imports
import torch
from torch import nn
from torchvision import datasets, transforms, models


def load_data_from_dir(data_dir_path):
    """ Loads and preprocesses the data. Expects path to a directory
        with train, valid, and test subfolders containing class examples.
        Return data dictionary with the preprocessed data.
    """

    # Creating default data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
                )
            ]),
        'test': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
                )
            ])
        }

    # Getting the images
    image_datasets = {
        x: datasets.ImageFolder(
                os.path.join(data_dir_path, x),
                data_transforms[x]
                )
        for x in ['train', 'valid', 'test']
        }

    # Creating data loaders to serve batches to the model
    dataloaders = {
        x: torch.utils.data.DataLoader(
                image_datasets[x],
                batch_size=64,
                shuffle=True
                )
        for x in ['train', 'valid', 'test']
        }

    # Saving helpful information about the datasets e.g. sizes, class names
    dataset_sizes = {
        x: len(image_datasets[x])
        for x in ['train', 'valid', 'test']
        }

    class_names = image_datasets['train'].classes

    return {
        'data_dir_path': data_dir_path,
        'image_datasets': image_datasets,
        'dataloaders': dataloaders,
        'dataset_sizes': dataset_sizes,
        'class_names': class_names
        }


def load_pretrained_model(arch='resnet18'):
    """ Loads user specified architecture. Currently supported:
        * resnet18
        * resnet152
    """

    if arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif arch == 'resnet152':
        model = models.resnet152(pretrained=True)
    else:
        raise Exception('Not able to recognize requested model architecture')
    return model


def replace_classifier(model, n_classes=102, hidden_units=300):
    """ Replaces the pretrained model's classifier with a custom
        classifier to fit the new dataset
    """

    # Freezing the pre-trained parameters
    for p in model.parameters():
        p.requires_grad = False

    # Building a new classifier and replacing the current
    clf = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(model.fc.in_features, hidden_units)),
                ('relu1', nn.ReLU()),
                ('d1', nn.Dropout(.2)),
                ('fc2', nn.Linear(hidden_units, hidden_units)),
                ('relu2', nn.ReLU()),
                ('d2', nn.Dropout(.2)),
                ('fc3', nn.Linear(hidden_units, hidden_units)),
                ('relu3', nn.ReLU()),
                ('d3', nn.Dropout(.2)),
                ('fc4', nn.Linear(hidden_units, n_classes)),
                ('output', nn.LogSoftmax(dim=1))
                ]))

    model.fc = clf


def train_model(model, optimizer, criterion, data_dict,
                num_epochs=2, calc_validation=True, device='cuda'):
    """ Main function for training the neural network. Across multiple epochs
        and batches, completes forward passes, calculates prediction accuracy,
        and completes backwards passes.
        Model is updated inplace.
    """

    since = time.time()

    print('*' * 25)
    print('Starting training process')

    model.to(device)  # Moving model to requested device
    phases = ['train', 'valid'] if calc_validation else ['train']

    # Iterating through epochs
    for epoch in tqdm(range(num_epochs)):
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Iterating through train and validation phase
        for phase in phases:
            print(phase)
            running_loss = 0
            running_corrects = 0
            model.train() if phase == 'train' else model.eval()

            # Iterating through batches in one epoch
            for inputs, labels in tqdm(
                                    data_dict['dataloaders'][phase],
                                    leave=False
                                    ):

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()  # stop grad from accumulating

                # Forward pass; only use autograd during training
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Update the optimizer
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Update running loss and num_correct for the current batch
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate stats for the current epoch and report to user
            epoch_loss = running_loss / data_dict['dataset_sizes'][phase]
            epoch_acc = running_corrects.double(
            ) / data_dict['dataset_sizes'][phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


#  Calculating accuracy on test set
def calc_test_accuracy(model, data_dict):
    ''' Calculates the test accuracy of the current model
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():  # Turning autograd tracking off
        for images, labels in tqdm_notebook(data_dict['dataloaders']['test']):

            images, labels = images.to(device), labels.to(
                device)    # Load images and labels into GPU
            # Forward prop
            outputs = model(images)
            # Get the label with the maximum likelihood
            _, predicted = torch.max(outputs.data, 1)
            # Get the number of processed data points
            total += labels.size(0)
            # Get the number of correct classified data points
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on all test images: %d %%' %
          (100 * correct / total))


def save_checkpoint(model, optimizer, criterion,
                    data_dict, arch, current_version):
    ''' Helper function to create and save a checkpoint for
        the current model.
    '''

    checkpoint = {'version': current_version,
                  'base_model': arch,
                  'classifier': model.fc,
                  'class_to_idx': data_dict['class_names'],
                  'optimizer': optimizer,
                  'criterion': criterion,
                  'classifier_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    torch.save(
        checkpoint,
        'checkpoint_{}_v{}.pth'.format(arch, current_version)
        )


def load_from_checkpoint(filepath, last_device='cuda:0'):
    ''' Helper function to load a checkpoint, generate the old model,
        and return all things neccesary to resume training
    '''

    checkpoint = torch.load(filepath, map_location=last_device)

    # Downloading the basemodel
    if checkpoint['base_model'] == 'resnet152':
        model = models.resnet152(pretrained=True)
        for p in model.parameters():
            p.requires_grad = False

    elif checkpoint['base_model'] == 'resnet18':
        model = models.resnet18(pretrained=True)
        for p in model.parameters():
            p.requires_grad = False

    else:
        print('Add base model to loading function')

    # Replacing the base model's classifier and updating state dict
    model.fc = checkpoint['classifier']
    model.load_state_dict(checkpoint['classifier_state_dict'])

    # Adding the class_to_idx dictionary to the model
    model.class_to_idx = checkpoint['class_to_idx']

    # Creating the optimizer and updateing state dict
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Creating the criterion used for training
    criterion = checkpoint['criterion']

    # Computing the current version of the checkpoint
    version = checkpoint['version']+1

    # Return tuple
    return model, criterion, optimizer, version


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Get dimensions for the image
    width = image.size[0]
    height = image.size[1]

    # Resizing the image while keeping aspect ratios intact
    proper_size = (256, int(256*height/width)
                   ) if width < height else (int(256*width/height), 256)
    im_resized = image.resize(proper_size)

    # Center cropping the image
    x_center = proper_size[0]/2
    y_center = proper_size[1]/2
    box = (x_center-112, y_center-112, x_center+112, y_center+112)
    im_cropped = im_resized.crop(box)

    # Normalize color channels
    im_numpy = np.array(im_cropped)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im_numpy_norm = (im_numpy - mean) / std

    # Transpose to move color dimesnion first
    im_numpy_norm_t = im_numpy_norm.transpose((2, 0, 1))

    return im_numpy_norm_t


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


pil_im = Image.open('flowers/train/85/image_04812.jpg')  # Wide example

print('Original image:')
plt.imshow(pil_im)
plt.show()

print('Processed imaged:')
p_im = process_image(pil_im)
imshow(p_im)
plt.show()

print('Checking dimensions:')
print(p_im.shape)


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained
        deep learning model.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # require no autograd for performance
    with torch.no_grad():

        # Load the orginal image
        pil_im = Image.open(image_path)
        # Change to numpy array and turn into tensor
        tensor_im = torch.Tensor(process_image(pil_im))
        # Add an additional dimension for batch size
        tensor_im.unsqueeze_(0)
        tensor_im = tensor_im.to(device)                  # Move image to GPU
        # Do forward prop and save the output
        output = model(tensor_im)
        # Get the k best precitions and loss
        nll, idx = output.topk(topk)
        # Cast to python lists
        nll, idx = nll[0].tolist(), idx[0].tolist()

        # Invert index for lookup
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        classes = [idx_to_class.get(item) for item in idx]

        # calculate probabilities
        proba = np.exp(nll)

        return proba, classes


def predict_single_image(image_path, model, class_names_json=None):
    ''' Helper function to predict a single image and output result as image
    '''
    probs, classes = predict(image_path, model)

    # If json lookup was provided use lookup name
    if class_names_json:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
            x_ticks = [cat_to_name[i] for i in classes]
    else:
        x_ticks = classes
    df = pd.DataFrame(probs, x_ticks, columns=['probability'])

    pil_im = Image.open(image_path)
    process_im = process_image(pil_im)

    # Creating subplots
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(7, 4))

    # Adding image to top plot
    imshow(process_im, ax=ax1)
    fig.suptitle(x_ticks[0], y=.95, fontsize=20)
    ax1.axis('off')

    # Adding probabilities to bottom plot
    ax2.set_aspect(1/6)
    df.plot.barh(ax=ax2, legend=False)
    plt.show()
