# Deep Learning Image Classifier

### About this project:
This Jupyter notebook illustrates the process of using deep neural networks and transfer learning to classify images of flowers. The project uses [PyTorch's ResNet152](https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet18) and Oxford University's [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) to achieve a 92% accuracy on the test set. The user can upload new images and get classification results through a command line interface.

![image classification example](../images/image_classification_example_1.png)


### How to get started:

1. The `Image Classifier Project.html` explains the process plus contains static images to help illustrate.

2. The `Image Classigier Project.ipynb` can be downloaded alongside the above mentioned image dataset and used to recreate the results. Unless you have access to a CUDA-enabled GPU the training process may take several hours to complete.

3. If you're not interested in the notebook format, but want to see the code as pure python, take a look at the `cli_application` folder. The two main scrips are `predict.py` and `train.py`. As mentioned above, re-training the classifier without GPU hardware takes several hours. The script automatically detects if a GPU is available and will make use of it, unless being told otherwise by the user. 
