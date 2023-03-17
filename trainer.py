import numpy as np
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from typing import Optional, Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from networks.core import FewShotClassifier


def compute_score(model, images, labels, loss_fn, optimizer, device):
    # Zero the gradient at each iteration
    optimizer.zero_grad()
    # Get the correct prediction scores
    scores = model(images.to(device))
    correct = (torch.max(scores.detach().data, 1)[1] == labels).sum().item()
    # Compute loss
    loss = loss_fn(scores, labels.to(device))
    # Backward propagation for calculating gradients
    loss.backward()
    # Update the weights
    optimizer.step()
    return loss, correct

def trainer(model: FewShotClassifier,
    data_loader: DataLoader,
    optimizer: Optimizer = None,
    loss_fn: Optional[Callable] = None,
    train: bool = True,
    training: str = 'episodic',
    verbose: bool = False,
    device: str = 'cpu'):
    
    ###### TRAINING ######
    if train:
        model.train()

        train_loss = []
        train_accuracy = []
        total_predictions = 0
        correct_predictions = 0

        if training == 'classical':
            with tqdm(data_loader, total=len(data_loader), desc="Classical Training") as tqdm_train:
                for images, labels in tqdm_train:
                    # Compute score and correct classifications
                    loss, correct = compute_score(model, images, labels, loss_fn, optimizer, device)
                    # Append accuracy and loss to lists
                    total_predictions += len(labels)
                    correct_predictions += correct
                    train_accuracy.append(correct_predictions / total_predictions)
                    train_loss.append(loss.item())
                    # Log loss in real time
                    tqdm_train.set_postfix(loss=np.mean(train_loss))
            return np.mean(train_loss), np.mean(train_accuracy)
        
        if training == 'episodic':
            with tqdm(enumerate(data_loader), total=len(data_loader), disable=not True, desc="Episodic Training") as tqdm_train:
                for i, (support_images, support_labels, query_images, query_labels, _) in tqdm_train:
                    model.process_support_set(support_images.to(device), support_labels.to(device))
                    # Compute score and correct classifications
                    loss, correct = compute_score(model, query_images, query_labels, loss_fn, optimizer, device)
                    # Append accuracy and loss to lists
                    total_predictions += len(query_labels)
                    correct_predictions += correct
                    train_accuracy.append(correct_predictions / total_predictions)
                    train_loss.append(loss.item())
                    # Log loss in real time
                    tqdm_train.set_postfix(loss=np.mean(train_loss)) 
            return np.mean(train_loss), np.mean(train_accuracy)
    
    ###### EVALUATING ######
    else: 
        labels = []
        predictions = []
        test_loss = []
        test_accuracy = []
        total_predictions = 0
        correct_predictions = 0

        model.eval()
        with tqdm(enumerate(data_loader), total=len(data_loader), disable=not True, desc="Evaluating") as tqdm_eval:
            for i, (support_images, support_labels, query_images, query_labels, _) in tqdm_eval:
                model.process_support_set(support_images.to(device), support_labels.to(device)) 
                scores = model(query_images.to(device)).detach()
                # Compute loss
                loss = loss_fn(scores, query_labels.to(device))
                correct = (torch.max(scores.detach().data, 1)[1] == query_labels).sum().item()
                # Get the predicted labels
                predicted_labels = torch.max(scores.data, 1)[1]
                labels += query_labels.tolist()
                predictions += predicted_labels.tolist()
                total_predictions += len(query_labels)
                correct_predictions += correct
                test_accuracy.append(correct_predictions / total_predictions)
                test_loss.append(loss.item())
                # Log accuracy in real time
                tqdm_eval.set_postfix(accuracy=correct_predictions / total_predictions)

        if verbose:
            print(metrics.classification_report(labels, predictions, digits=3))

        return np.mean(test_loss), np.mean(test_accuracy)


    
