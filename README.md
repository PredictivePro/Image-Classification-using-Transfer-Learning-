# Image-Classification-using-Transfer-Learning

This project demonstrates the process of image classification using transfer learning with a focus on the popular ResNet-50 architecture pre-trained on the ImageNet dataset. Transfer learning allows us to leverage the knowledge learned from a large dataset (ImageNet) and adapt it for a new task (MNIST digit classification).

## Overview
In this project, we use PyTorch and torchvision to build an image classifier that can recognize handwritten digits from the MNIST dataset. We take advantage of the ResNet-50 model, fine-tuning it for our specific task. The project consists of several components:

1. Importing Libraries: We start by importing the necessary Python libraries, including NumPy, PyTorch, and torchvision, to help us with data handling, model training, and evaluation.

2. Configuration Parameters: We define some configuration parameters such as the number of training epochs, batch size, and learning rate.

3. Device Selection: The code checks if a CUDA-compatible GPU is available and selects it for faster training; otherwise, it falls back to CPU.

4. Model Selection: We provide an option to either train the model with random weights or use pre-trained weights from ResNet-50. This flexibility allows you to experiment with different initialization strategies.

5. Data Preparation: The MNIST dataset is loaded and preprocessed using torchvision's built-in functionalities. We also create a target dataset for training to adjust the dataset size for demonstration purposes.

6. Loss and Optimizer: We define the loss function (Cross-Entropy Loss) and the optimizer (Stochastic Gradient Descent) for model training.

7. Training: The training loop iterates over the specified number of epochs, computing loss and optimizing model parameters. It prints training progress, including loss and accuracy.

8. Testing: After training, the model's performance is evaluated on a separate test dataset. We calculate and print the classification accuracy.

9. Timing Information: We record and display the time taken for training the model.

### How to Use
Clone the Repository: Clone this GitHub repository to your local machine.

Set Up the Environment: Make sure you have Python and PyTorch installed. 
You can install the required libraries using pip:

Example: 

pip install torch torchvision

You can also download the MNIST dataset from here: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Run the Code: Execute the provided Python script to start the image classification process. You can adjust configuration parameters like the number of epochs, batch size, optimizers, and learning rate in the code.

Experiment: Feel free to experiment with different hyperparameters, try other pre-trained models, or even modify the model architecture to suit your specific image classification task.

Observe Results: Check the training progress and final accuracy on the test dataset. The training time will also be displayed.

#### Conclusion
This project demonstrates the power of transfer learning in image classification tasks, using the ResNet-50 model as a strong starting point. By fine-tuning the model on a different dataset (MNIST), we achieve impressive accuracy. You can use this code as a foundation for more complex image classification tasks or as a learning resource to understand transfer learning in deep learning.
