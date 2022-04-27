from xmlrpc.client import boolean
from EDU import EDU
from Transition_system import Arc_eager
from models import BertArcNet, SEQ_LEN, NaiveBertArcNet
from UAS_parsing import assembled_sentence_execution, wrapper_model, modify_contextualized_embeddings
import pickle
from torch.utils.data import DataLoader, TensorDataset
import torch
import copy
import random
import numpy as np
from transformers import AutoTokenizer, AutoModel
from relation_labeling import build_relation_list, assembled_transform_heads
from training_utils import build_in_sentence_data, build_between_sentence_data,\
build_training_data, classification_train, classification_validate
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse


def main():
    #parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", default="bert-base-uncased", type=str)
    parser.add_argument("--model_name", default="bert-base-uncased", type=str)
    parser.add_argument("--dataset", default="scidtb", type=str)
    parser.add_argument("--path_train_data", default='preprocessed_data/sci_train.data', type=str)
    parser.add_argument("--path_dev_data", default='preprocessed_data/sci_dev.data', type=str)
    #in_sentence and between_sentence options
    parser.add_argument("--train_option", default='in_sentence', type=str)
    #the path to save the model
    parser.add_argument("--dest_path", default='temp_model.pt', type=str)
    parser.add_argument("--learning_rate", default=2e-5, type=np.float32)
    parser.add_argument("--batch_size", default=24, type=np.int32)
    parser.add_argument("--epochs", default=3, type=np.int32)
    parser.add_argument("--verbose", default=True, type=np.bool8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    bert = AutoModel.from_pretrained(args.model_name)

    with open(args.path_train_data,'rb') as fb:
        train_data = pickle.load(fb)
    with open(args.path_dev_data,'rb') as fb:
        dev_data = pickle.load(fb)    

    train_in_sentence_data = build_in_sentence_data(train_data)
    dev_in_sentence_data = build_in_sentence_data(dev_data)
    train_between_sentence_data = build_between_sentence_data(train_data)
    dev_between_sentence_data = build_between_sentence_data(dev_data)

    if args.train_option == 'in_sentence':
        for edus in train_in_sentence_data:
            edus = modify_contextualized_embeddings(edus, tokenizer, SEQ_LEN = SEQ_LEN)
        for edus in dev_in_sentence_data:
            edus = modify_contextualized_embeddings(edus, tokenizer, SEQ_LEN = SEQ_LEN)
        train_features, train_labels = build_training_data(train_in_sentence_data)
        dev_features, dev_labels = build_training_data(dev_in_sentence_data)
    elif args.train_option == 'between_sentence':
        for edus in train_between_sentence_data:
            edus = modify_contextualized_embeddings(edus, tokenizer, SEQ_LEN = SEQ_LEN)
        for edus in dev_between_sentence_data:
            edus = modify_contextualized_embeddings(edus, tokenizer, SEQ_LEN = SEQ_LEN)
        train_features, train_labels = build_training_data(train_between_sentence_data)
        dev_features, dev_labels = build_training_data(dev_between_sentence_data)
    else:
        raise Exception('Train option undefined - supported train options: in_sentence, between_sentence')
    torch.cuda.empty_cache()
    #set the right format for training data and test data, for pytorch
    X_train = train_features.long()
    X_dev = dev_features.long()
    y_train = torch.Tensor(train_labels).long()
    y_dev = torch.Tensor(dev_labels).long()
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last = True)
    dev_dataset = TensorDataset(X_dev, y_dev)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, drop_last = True)
    # trainer = pl.Trainer(devices = "auto", accelerator = 'gpu',\
    #         auto_scale_batch_size= "power" , max_epochs = 3, benchmark = True,auto_lr_find=True)
    # model = BertArcNetPL(bert)
    # trainer.fit(model, train_dataloader, dev_dataloader)
    model = BertArcNet(bert).cuda()
    # model = torch.load('trained_models/SciDTB/sciDTB_between_sentence.pt')
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    torch.cuda.empty_cache()
    for i in range(args.epochs):
        if args.verbose:
            print(f'for the {i}th epoch')
        train_losses.append(classification_train(model, train_dataloader, optimizer, criterion,  verbose = args.verbose))
        with torch.no_grad():
            val_losses.append(classification_validate(model, dev_dataloader, criterion,verbose = args.verbose)) 
    torch.save(model, args.dest_path)

if __name__ == "__main__":
    main()
