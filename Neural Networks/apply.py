# Credits : Stephanie, Käs - RWTH Aachen 
#           for plot_multiple, visualise_dataset and train_classifier functions

LOG_ROOT = "tensorboard_logs"
USE_GPU = True  # Set to True if you have installed tensorflow for GPU


import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from Softmax import softmax_regression
from Softmax_Adam import softmax_regression_adam
from Multi_Layered_Perceptron import tanh_mlp
from Data_Augmentation import *
from Relu import relu_mlp
from CNN import cnn
from Forward_Pass import *
from Batch_Norm import cnn_batchnorm
from Strided_Conv import cnn_strides
from Global_Pooling import *
from Resnet import *


# x, y = train_data[0]  # get an example from the dataset
# print(f"\nShape of an image: {x.shape}.")
# visualize_dataset(train_data, labels)
    
# Train a model with Softmax Regression

def train_classifier(
    model,
    opt,
    logdir,
    train_data=train_data,
    test_data=test_data,
    batch_size=128,
    n_epochs=1,
    lr_scheduler=None,
):
    writer = SummaryWriter(f"{LOG_ROOT}/{logdir}-{time.strftime('%y%m%d_%H%M%S')}")
    layout = {
        "Losses": {"losses": ["Multiline", ["loss/train", "loss/test"]]},
        "Accuracy": {"accuracy": ["Multiline", ["accuracy/train", "accuracy/test"]]},
    }
    writer.add_custom_scalars(layout)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=6)

    criterion = nn.CrossEntropyLoss()

    start = time.time()

    for epoch in range(n_epochs):
        sample_count = 0
        loss_sum = 0
        correct = 0
        n_batches = len(train_loader)
        model.train()
        for i, (xs, ys) in enumerate(train_loader):
            if USE_GPU:
                xs = xs.cuda()
                ys = ys.cuda()
            out = model(xs)
            loss = criterion(out, ys)
            loss.backward()
            opt.step()
            opt.zero_grad()

            loss_sum += loss.item() * xs.shape[0]
            _, pred = torch.max(out, 1)
            correct += (pred == ys).sum().item()
            sample_count += xs.shape[0]
            print(f"Train epoch {epoch+1}, step {i+1}/{n_batches}", end="    \r")

        train_loss = loss_sum / sample_count
        train_accuracy = correct / sample_count

        with torch.no_grad():  # do not store gradients during testing, decreases memory consumption
            sample_count = 0
            loss_sum = 0
            correct = 0
            n_batches = len(test_loader)
            model.eval()
            for i, (xs, ys) in enumerate(test_loader):
                if USE_GPU:
                    xs = xs.cuda()
                    ys = ys.cuda()
                out = model(xs)
                loss = criterion(out, ys)
                loss_sum += loss.item() * xs.shape[0]
                _, pred = torch.max(out, 1)
                correct += (pred == ys).sum().item()
                sample_count += xs.shape[0]
                print(f"Test epoch {epoch+1}, step {i+1}/{n_batches}", end="    \r")

            test_loss = loss_sum / sample_count
            test_accuracy = correct / sample_count

        writer.add_scalar("loss/train", train_loss, epoch + 1)
        writer.add_scalar("accuracy/train", train_accuracy, epoch + 1)
        writer.add_scalar("loss/test", test_loss, epoch + 1)
        writer.add_scalar("accuracy/test", test_accuracy, epoch + 1)

        if lr_scheduler is not None:
            lr_scheduler.step()
            writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch + 1)

        print(
            f"Epoch {epoch+1} | train loss: {train_loss:.3f}, train accuracy: {train_accuracy:.3f}, "
            + f"test loss: {test_loss:.3f}, test accuracy: {test_accuracy:.3f}, "
            + f"time: {str(datetime.timedelta(seconds=int(time.time()-start)))}"
        )
  
  
if __name__ == "__main__":  
    # Train the softmax classifier
    opt = optim.SGD(softmax_regression.parameters(), lr=1e-2)
    train_classifier(softmax_regression, opt, "softmax_regression")
    
    # One of the most popular vairant developed to improve the stochastic gradient model is the 
    # Adam:"https://arxiv.org/abs/1412.6980, "adaptive moment estimation.
    
    # Train the Softmax Adam classifier
    opt = optim.Adam(softmax_regression_adam.parameters(), lr=2e-4)
    train_classifier(softmax_regression_adam, opt, "softmax_regression_adam")
    
    # Multiplication by the weights  𝑊 can be interpreted as computing responses to correlation templates per image class.
    # That means, we can reshape the weight array  𝑊 to a obtain "template images".
    W = softmax_regression[1].weight.data
    templates = W.reshape(10, 3, 32, 32)
    
    # We normalize the templates for visualization
    mini = templates.min()
    maxi = templates.max()
    rescaled_templates = (templates - mini) / (maxi - mini)
    plot_multiple(rescaled_templates.cpu(), labels, max_columns=5)

    # Train the tanh MLP 
    opt = optim.Adam(tanh_mlp.parameters(), lr=2e-4)
    train_classifier(tanh_mlp, opt, f"tanh_mlp")
        
    # Data Augmentation
    # Train the tanh MLP with augmented data
    opt = optim.Adam(tanh_mlp.parameters(), lr=2e-4)
    train_classifier(tanh_mlp, opt, f"tanh_mlp_augmented", train_data=augmented_train_data)
    
    # ReLu
    # Train ReLU
    opt = optim.Adam(relu_mlp.parameters(), lr=2e-4)
    train_classifier(relu_mlp, opt, "relu_mlp", train_data=augmented_train_data)
    
    # CNN
    # Train CNN 
    opt = optim.Adam(cnn.parameters(), lr=1e-3)
    train_classifier(cnn, opt, "cnn", train_data=augmented_train_data)
    
    # Implementing the Forward Pass
    my_logits = my_predict_cnn(inp_np, W1, b1, W2, b2, W3, b3)
    pytorch_logits = cnn(inp[np.newaxis])[0]
    if np.mean((my_logits - pytorch_logits.detach().cpu().numpy()) ** 2) > 1e-5:
        print("Something isn't right! PyTorch gives different results than my_predict_cnn!")
    else:
        print("Congratulations, you got correct results!")
    
    
    # Batch Normalisation
    # Train the CNN with Batch Norm
    opt = optim.Adam(cnn_batchnorm.parameters(), lr=1e-3)
    train_classifier(cnn_batchnorm, opt, "cnn_batchnorm", train_data=augmented_train_data)
    
    
    # Strided Convolutions
    # Train the CNN with Strided Convolutions
    opt = optim.Adam(cnn_strides.parameters(), lr=1e-3)
    train_classifier(cnn_strides, opt, "cnn_strides", train_data=augmented_train_data)
    
    
    # Global Pooling
    # Train the CNN with global pooling
    opt = optim.Adam(cnn_global_pool.parameters(), lr=1e-3)
    train_classifier(cnn_global_pool, 
                     opt, 
                     logdir="cnn_global_pool", 
                     train_data=augmented_train_data)
    
    
    # Resnet 
    # Train the Classifier
    opt = optim.Adam(resnet.parameters(), lr=1e-3, weight_decay=1e-4)
    train_classifier(resnet, opt, f"resnet", train_data=augmented_train_data)
    
    
    # Learning Rate Decay
    resnet_decay = ResNet()
    if USE_GPU:
        resnet_decay.cuda()
        
    opt = optim.Adam(resnet_decay.parameters(), lr=1e-3, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.MultiStepLR(opt, [35, 45], gamma=0.1)
    train_classifier(resnet_decay, 
                     opt,
                     "resnet_decay",
                     lr_scheduler=scheduler,
                     train_data=augmented_train_data,)
    
    plt.show()
