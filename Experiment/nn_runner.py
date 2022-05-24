import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import time

from adversarial import *


def eval_net(model, criterion, eval_loader, device):
    batch_history = {
        "loss": [],
        "accuracy": []
    }
    
    # Initiate evaluation mode
    model.eval()
    # Leave gradients unchanged
    # to avoid re-training
    eval_loss, eval_accuracy = 0, 0
    with torch.no_grad():
        # Iterate over the dataset
        for image, label in eval_loader:
            # Push data onto the available device
            image = image.to(device)
            label = label.to(device)
            # Make predictions
            output = model(image)
            # Calculate batch loss
            batch_loss = criterion(output, label)
            # Update total validation loss
            eval_loss += batch_loss.item() * image.size(0)
            # Find out how many we got right
            correct_results = (output.argmax(dim=1) == label)
            # Update total accuracy
            batch_acc = correct_results.sum().item()
            eval_accuracy += batch_acc
            batch_history['loss'].append(batch_loss / len(eval_loader.sampler))
            batch_history['accuracy'].append(batch_acc / len(eval_loader.sampler))
        # Normalize loss and accuracy
        eval_loss = eval_loss / len(eval_loader.sampler)
        eval_accuracy = eval_accuracy / len(eval_loader.sampler)
        print(75 * "=")
        print('Evaluation loss: {:.5f} - Evaluation accuracy: {:.2f} %'.format(eval_loss, eval_accuracy * 100))
        
    return eval_loss, eval_accuracy, batch_history


def train_net(model, optimizer, criterion, train_loader, device, adversary):
    train_loss, train_accuracy = 0, 0
    
    # Iterate over the dataset
    for image, label in train_loader:
        # Push data onto the available device
        image = image.to(device)
        label = label.to(device)
        # Re-initialize the gradients
        optimizer.zero_grad()
        # Implicitly call the forward method
        output = model(image)
        # Calculate loss for this batch
        batch_loss = criterion(output, label)
        # Perform backpropagation
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        batch_loss.backward()
        # Update weights
        optimizer.step()
        # Update total loss
        train_loss += batch_loss.item() * image.size(0)
        # Calculate training accuracy for this batch
        correct_results = (output.argmax(dim=1) == label)
        # Update total train accuracy
        train_accuracy += correct_results.sum().item()
        
        if adversary is not None:
            adv_image = adversary.perturbe(image, label)
            # Re-initialize the gradients
            optimizer.zero_grad()
            # Implicitly call the forward method
            output = model(adv_image)
            # Calculate loss for this batch
            batch_loss = criterion(output, label)
            # Perform backpropagation
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            batch_loss.backward()
            # Update weights
            optimizer.step()

    # Normalize loss and accuracy
    train_loss = train_loss / len(train_loader.sampler)
    train_accuracy = train_accuracy / len(train_loader.sampler)
   
    print("\n" + 75 * "=")
    print('Train loss: {:.5f} - Train accuracy: {:.2f} %'.format(train_loss, train_accuracy * 100))

    return train_loss, train_accuracy


def fit (
    model, 
    optimizer, 
    criterion, 
    epochs, 
    train_loader, 
    valid_loader=None, 
    device=torch.device('cpu'), 
    adversary=None
):
    # Push model 
    # to GPU or CPU
    model.to(device)

    train_history = {
        'loss': [],
        'accuracy': []
    }

    valid_history = {
        'loss': [],
        'accuracy': []
    }
    
    start_time = time.time()

    for _ in tqdm(range(epochs)):
        train_loss, train_accuracy = train_net(model, optimizer, criterion, train_loader, device, adversary)
        
        # Log history of loss and accuracy
        train_history['loss'].append(train_loss)
        train_history['accuracy'].append(train_accuracy)

        if valid_loader is not None:
            eval_loss, eval_accuracy, _ = eval_net(model, criterion=criterion, eval_loader=valid_loader, device=device)
            # Log evaluation history
            valid_history['loss'].append(eval_loss)
            valid_history['accuracy'].append(eval_accuracy)
            
        print('=' * 75 + "\n")
        
    total_time = (time.time() - start_time) / 60

    return train_history, valid_history, total_time
    
    
def run(train_loader, valid_loader, test_loader, args):
    is_cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda_available else 'cpu')

    model = eval(str('models.' + args["model"]))()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    adversary = None
    if args["noise_type"] == "fgsm":
        adversary = FGSM(model, criterion, device, args["epsilon"], args["iters"], args["alpha"])
    elif args["noise_type"] == "pgd":
        adversary = PGD(model, criterion, device, args["epsilon"], args["iters"], args["alpha"])
    
    _, _, total_time = fit (
        model, optimizer, criterion, 
        epochs=args["epochs"], train_loader=train_loader,
        valid_loader=valid_loader, device=device,
        adversary=adversary
    )
    
    torch.save(model.state_dict(), args["model"] + "_" + args["noise_type"] + ".pth")
       
    loss, acc, batch = eval_net(model=model, criterion=criterion, eval_loader=test_loader, device=device)
    
    return loss, acc, batch, total_time