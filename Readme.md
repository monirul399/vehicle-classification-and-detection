# Vehicle Classification and Detection with Transfer Learning

Vehicle classification and detection has been a vital field of application for deep learning and image processing techniques, playing a crucial role in intelligent transport management and AI-assisted driving systems. This repository presents a vehicle classification and detection system designed to detect and classify low-speed and high-speed vehicles in Bangladesh.

## Overview

In this project, we have implemented and evaluated the performance of 11 pre-trained deep convolutional neural network (CNN) models: YOLOv8, MobileNetV2, GoogLeNet, AlexNet, ResNet-50, SqueezeNet, VGG19, DenseNet-121, Xception, InceptionV3, and NASNetMobile. These models were tested on six vehicle classification and detection datasets: BIT-Vehicle, IDD, DhakaAI, Poribohon-BD, Sorokh-Poth, and VTID2.

Our findings indicate that the YOLOv8, MobileNetV2, and GoogLeNet models outperform the other models in terms of accuracy and performance.

## Repository Structure

- `data/`: Contains the datasets used for training and evaluation.
- `notebooks/`: Jupyter Notebooks for data exploration, preprocessing, model training, and evaluation.
- `results/`: Stores the model outputs, evaluation metrics, and visualizations.
- `requirements.txt`: Lists the required Python packages and their versions.

## Installation

To get started with a repository, you'll need [Python 3](https://www.python.org/) and [Kaggle](https://www.kaggle.com/).

I would recommend you create a virtual environment in the current directory. Any libraries you download (such as numpy) will be placed there. Enter the following into a command prompt:


```bash
python3 -m venv venv
```

This creates a virtual environment in the `venv` directory. To activate it:

```bash
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\scripts\activate
```

With your virtual environment active, navigate to the project directory and run the following command to install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
This command will install all the packages listed in the `requirements.txt` file, along with their dependencies.


| Dataset | Models | Notebook |
|----------|----------|----------|
| DhakaAI    | YOLOv8   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](url)   |
|          | MobileNetV2   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/nasnetmobile-on-dhakaai.ipynb)   |
|          | GoogLeNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/googlenet-on-dhakaai.ipynb)   |
|          | AlexNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/alexnet-on-dhakaai.ipynb)   |
|          | ResNet-50   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/resnet50-on-dhakaai.ipynb)   |
|          | SqueezeNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/squeezenet-on-dhakaai.ipynb)   |
|          | VGG-19   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/vgg19-on-dhakaai.ipynb)   |
|          | DenseNet-121   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/densenet121-on-dhakaai.ipynb)   |
|          | Xception   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/xception-on-dhaka-ai.ipynb)   |
|          | InceptionV3  | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/inceptionv3-on-dhakaai.ipynb)  |
|          | NASNetMobile   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/nasnetmobile-on-dhakaai.ipynb)   |
| VTID2    | YOLOv8   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](url)   |
|          | MobileNetV2   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/mobilenet-on-vtid2.ipynb)   |
|          | GoogLeNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/googlenet-on-vtid2.ipynb)   |
|          | AlexNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/alexnet-on-vtid2.ipynb)   |
|          | ResNet-50   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/resnet50-on-vtid2.ipynb)   |
|          | SqueezeNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/sueezenet-on-vtid2.ipynb)   |
|          | VGG-19   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/vgg19-on-vtid2.ipynb)   |
|          | DenseNet-121   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/densenet121-on-vtid2.ipynb)   |
|          | Xception   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/xception-vtid2.ipynb)   |
|          | InceptionV3  | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/inceptionv3-on-vtid2.ipynb)  |
|          | NASNetMobile   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/nasnetmobile-on-vtid2.ipynb)   |
| BITVehicle    | YOLOv8   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](url)   |
|          | MobileNetV2   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/mobilenet-on-bit-vehicle.ipynb)   |
|          | GoogLeNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/googlenet-on-bit-vehicle.ipynb)   |
|          | AlexNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/alexnet-on-bit-vehicle.ipynb)   |
|          | ResNet-50   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/resnet50-on-bit-vehicle.ipynb)   |
|          | SqueezeNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/sueezenet-on-bit-vehicle.ipynb)   |
|          | VGG-19   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/vgg19-on-bit-vehicle.ipynb)   |
|          | DenseNet-121   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/densenet121-on-bit-vehicle.ipynb)   |
|          | Xception   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/xception-on-bit-vehicle.ipynb)   |
|          | InceptionV3  | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/inceptionv3-on-bit-vehicle.ipynb)  |
|          | NASNetMobile   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/nasnetmobile-on-bit-vehicle.ipynb)   |
| IDD    | YOLOv8   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](url)   |
|          | MobileNetV2   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/mobilenet-on-idd.ipynb)   |
|          | GoogLeNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/googlenet-on-idd.ipynb)   |
|          | AlexNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/alexnet-on-vtid2.ipynb)   |
|          | ResNet-50   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/resnet50-on-idd.ipynb)   |
|          | SqueezeNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/sueezenet-on-idd.ipynb)   |
|          | VGG-19   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/vgg19-on-idd.ipynb)   |
|          | DenseNet-121   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/densenet121-on-idd.ipynb)   |
|          | Xception   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/xception-on-idd.ipynb)   |
|          | InceptionV3  | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/inceptionv3-on-idd.ipynb)  |
|          | NASNetMobile   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/nasnetmobile-on-idd.ipynb)   |
| PoribohonBD    | YOLOv8   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](url)   |
|          | MobileNetV2   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/mobilenet-on-poribohonbd.ipynb)   |
|          | GoogLeNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/googlenet-on-poribohonbd.ipynb)   |
|          | AlexNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/alexnet-on-poribohonbd.ipynb)   |
|          | ResNet-50   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/resnet50-on-poribohon-bd.ipynb)   |
|          | SqueezeNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/sueezenet-on-poribohonbd.ipynb)   |
|          | VGG-19   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/vgg19-on-poribohon-bd.ipynb)   |
|          | DenseNet-121   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/densenet121-on-poribohon-bd.ipynb)   |
|          | Xception   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/xception-pribohonbd.ipynb)   |
|          | InceptionV3  | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/inceptionv3-on-poribohon-bd.ipynb)  |
|          | NASNetMobile   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/nasnetmobile-on-poribohon-bd.ipynb)   |
| Sorokh-Poth    | YOLOv8   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](url)   |
|          | MobileNetV2   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/mobilenet-on-sorokh-poth.ipynb)   |
|          | GoogLeNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/googlenet-on-sorokh-poth.ipynb)   |
|          | AlexNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/alexnet-on-sorokh-poth.ipynb)   |
|          | ResNet-50   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/resnet50-on-sorokhpoth.ipynb)   |
|          | SqueezeNet   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/sueezenet-on-sorokh-poth.ipynb)  |
|          | VGG-19   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/xception-on-sorokh-poth.ipynb)   |
|          | DenseNet-121   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/densenet121-on-sorokhpoth.ipynb)   |
|          | Xception   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/xception-on-sorokh-poth.ipynb)   |
|          | InceptionV3  | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/inceptionv3-on-sorokhpoth.ipynb)  |
|          | NASNetMobile   | [![Open Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/monirul399/vehicle-classification-and-detection/blob/main/notebooks/mobilenet-on-sorokh-poth.ipynb)   |
