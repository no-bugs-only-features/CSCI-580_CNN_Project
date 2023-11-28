# CNN-project
[![Python Lint](https://github.com/nickhib/CNN-project/actions/workflows/py_lint_check.yml/badge.svg?branch=main)](https://github.com/nickhib/CNN-project/actions/workflows/py_lint_check.yml) <br>

## Goal:
The goal of this project is to train a convolutional neural network to correctly classify handwritten digits.

## Data:
The dataset used for this project will come from MNIST (http://yann.lecun.com/exdb/mnist/) which provides a dataset with training images and their labels, as well as a dataset with test images and their correct labels.

## Algorithm:
We intend to use a convolutional neural network to solve this problem. 

## Tools:
We will implement this project in Python using the TensorFlow library’s implementation of a CNN. 

## Using Docker:
### Build and Run the Container
```
docker build -t cnn-project .
docker run -v "$(pwd)":/cnn_project -it cnn-project
```

Now you're in the Docker container!<br>
Run "python test.py" to verify everything is working.<br>
Type "exit" to leave container.<br>

You can also use wsl and docker desktop to get the container going.<br>
Type "code ." to open vscode in the current directory.<br>

### Compile and Test a CNN Model
```
python your_model_file.py
python test_cnn.py your_cnn_model.model
```
## docker display
please follow link for instructions<br>
```
https://drive.google.com/file/d/1dLsgAz2mUjaHWR84iS1ZfE2NyiX7jqlc/view?usp=sharing
```

## Group Members:
- Eric Galvan
- Kendal Hasek
- Nicholas Hibbard
- Chloe Koschnick
- Dylan Wright

## Citations:
This project was inspired by a video tutorial entitled **Neural Network Python Project - Handwritten Digit Recognition**, created by **NeuralNine**. You can watch the video [here](https://www.youtube.com/watch?v=bte8Er0QhDg).
