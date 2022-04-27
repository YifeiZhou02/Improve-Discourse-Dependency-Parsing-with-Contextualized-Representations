import random
import numpy as np
import copy
import pickle
from transformers import AutoModel
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from Transition_system import Arc_eager

def build_in_sentence_data(data):
    """
    split the paragraph into sentences
    return a list of all sentences of EDUs
    """
    in_sentence_data = []
    for edus in data:
        in_sentence_data.append([])
        n = len(edus)
        for j, edu in enumerate(edus):
            in_sentence_data[-1].append(edu)
            if j < n -1 and edus[j].sentenceNo != edus[j+1].sentenceNo:
                in_sentence_data.append([])
    return in_sentence_data
def build_between_sentence_data(data):
    """
    extract the inter-sentential level parsing
    return a list
    each element in the list is a list of root EDUs of each sentences in a 
    discourse
    """
    between_sentence_data = []
    for edus in data:
        between_sentence_data.append([])
        for edu in edus:
            if edu.head == 0 or edus[edu.head-1].sentenceNo != edu.sentenceNo:
                between_sentence_data[-1].append(edu)
    return between_sentence_data

def build_training_data(data):
    """
    build static oracle for action classification, using the saved 
    attribute embedding in the each EDU
    """
    features = []
    #the four binary label for action
    ys = []
    for edus in data:
        trans = Arc_eager(edus)
        action = trans.golden_decide()
        while(action!=-1):
            features.append(trans.get_feature())
            ys.append(action)
            trans.move(action)
            action = trans.golden_decide()
    features = torch.cat(features)
    return features,ys

def classification_train(net, train_dataloader, optimizer, criterion, verbose = False):
    """
    training the UAS parser with offline logged data
    it can also be used for finetuning bert model for relation prediction
    """
    net = net.cuda()
    losses = []
    weights = []
    for batch_idx, (inputs, target) in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()
        outputs = net(inputs.cuda())
        loss = criterion(outputs, target.cuda())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        weights.append(len(target))
    losses = np.array(losses)
    weights = np.array(weights)
    result = np.sum(losses*weights)/np.sum(weights)
    if verbose:
        print(f'the training loss is {result}')
    return result
#testing for models with BERT layers
def classification_validate(net, val_dataloader, criterion, verbose = False):
    """
    validating the UAS parser with offline logged data
    it can also be used for finetuning bert model for relation prediction
    """
    net = net.eval().cuda()
    losses = []
    weights = []
    accuracies = []
    for batch_idx, (inputs, target) in enumerate(val_dataloader):
        outputs = net(inputs.cuda()).cpu()
        target = target.cpu()
        accuracy = torch.sum(torch.argmax(outputs, dim = 1) == target)
        accuracies.append(accuracy)
        loss = criterion(outputs, target)
        losses.append(loss.item())
        weights.append(len(target))
    losses = np.array(losses)
    weights = np.array(weights)
    accuracies = np.array(accuracies)
    val_result = np.sum(losses*weights)/np.sum(weights)
    if verbose:
        print(f'the validation loss is {val_result}')
        print(f'the validation accuracy is {np.sum(accuracies)/np.sum(weights)}')
    return val_result