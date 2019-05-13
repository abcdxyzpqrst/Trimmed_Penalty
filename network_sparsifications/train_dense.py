import time
import math
import numpy as np
import torch
import argparse
import torch.nn.functional as F
from torch import nn
from models import TrimMLP
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

def accuracy(model, valid_loader):
    correct = 0.0
    for images, labels in valid_loader:
        images = images.view(-1, 28*28)
        if torch.cuda.is_available():
            images, labels = images.cuda(async=True), labels.cuda(async=True)

        output = model(images)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
    return 100. * correct / len(valid_loader.dataset)

def main():
    # set random seed for reproducibility
    torch.manual_seed(12345678)

    # argument parsing
    parser = argparse.ArgumentParser(description="Neuron-Pruned MLP with the Trimmed ℓ₁ Penalty on Network Parameters")
    parser.add_argument("--batch_size", type=int, default=100, 
                        help="minibatch size")
    parser.add_argument("--max_epochs", type=int, default=200,
                        help="maximum number of epochs for post-train")
    parser.add_argument("--hidden", type=int, nargs='+', 
                        help="hidden size of MLP")
    parser.add_argument("--alphas", type=float, nargs='+', 
                        help="belief of sparsity level for each layer")
    parser.add_argument("--lambdas", type=float, nargs='+', 
                        help="regularization parameters for each layer")
    parser.add_argument("--eta", type=float, default=0.001, 
                        help="MLP parameter learning rate")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Trimming parameter learning rate")
    parser.add_argument("--bias", type=str, default="False",
                        help="Whether or not to use bias mask (if bias mask is used, the activation goes to exact zero)")
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="enables CUDA GPU training")
    
    # learning information configuration
    args = parser.parse_args()
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    hidden = args.hidden
    alphas = args.alphas
    lambdas = args.lambdas
    eta = args.eta
    tau = args.tau
    task = "\n#############################################################################################\n\nNeuron-Pruned MLP with the Trimmed ℓ₁-penalty on MNIST handwritten digit recognition problem\n\n#############################################################################################\n"
    print (task)
    
    if args.bias == "True":
        bias = True
    else:
        bias = False

    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
            batch_size=batch_size, shuffle=True, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
            batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # pruning with the Trimmed \ell_1 Regularization
    model = TrimMLP(784, 10, hidden_size=hidden, lambdas=lambdas, alphas=alphas, eta=eta, tau=tau)
    if torch.cuda.is_available():
        model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
   
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)
    
    generalization_error = []
    trim_epochs = []

    for epoch in range(max_epochs):
        start_time = time.time()
        
        # Train mode
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28*28)
            if torch.cuda.is_available():
                images, labels = images.cuda(async=True), labels.cuda(async=True)
            output = model(images)
            loss = criterion(output, labels)
            loss += model.reg_theta_loss()
            loss += model.kld()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        # parameter update

            model.trimmed_l1_regularization()
            model.set_mask(bias)
            model.apply_mask()
        end_time = time.time()
        
        # Test mode
        model.eval()
        trim_loss = model.trimming_loss()
        test_error = round(100 - accuracy(model, valid_loader), 2)
        
        if trim_loss == 0.0:
            generalization_error.append(test_error)
            trim_epochs.append(epoch+1)
        else:
            pass
        
        print ("Epoch: {}\nTraining Loss: {}\nTrimming Loss: {}\nGeneral Error: {}%\nArchitectures: {}\nElapsed Times: {} (sec)\n".format(epoch+1, loss, trim_loss, test_error, model.architecture(), round(end_time - start_time, 2)))
    
    minimum_error = min(generalization_error)
    generalization_error = np.array(generalization_error)
    idx = generalization_error.argmin()
    print ("Test error: {}%".format(min(generalization_error)))
    print ("Best epoch: {}".format(trim_epochs[idx]))
if __name__ == "__main__":
    main()
