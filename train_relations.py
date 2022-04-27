from utils.EDU import EDU
from utils.Transition_system import Arc_eager
from models.models import BertArcNet, SEQ_LEN, NaiveBertArcNet, BertRelationNet, RelationLSTMTagger, RelationMLPTagger
from tqdm import tqdm
from utils.UAS_parsing import assembled_sentence_execution, wrapper_model, modify_contextualized_embeddings
import pickle
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from utils.relation_labeling import build_relation_list, assembled_transform_heads
from utils.training_utils import build_in_sentence_data, build_between_sentence_data,\
build_training_data, classification_train, classification_validate
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from utils.relation_labeling import prepare_finetune_dataloader, LSTMDataset,\
    prepare_seq_label_dataloader, labeling_train, labeling_validate
# bert = AutoModel.from_pretrained("bert-base-chinese")


def main():
    #parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", default="bert-base-uncased", type=str)
    #the name of the transformer, default is bert-base-uncased
    parser.add_argument("--model_name", default="bert-base-uncased", type=str)
    parser.add_argument("--dataset", default="scidtb", type=str)
    parser.add_argument("--path_train_data", default='preprocessed_data/sci_train.data', type=str)
    parser.add_argument("--path_dev_data", default='preprocessed_data/sci_dev.data', type=str)
    #train options are in_sentence_bert, in_sentence_lstm, between_sentence_bert, and between_sentence_lstm
    parser.add_argument("--train_option", default='in_sentence_bert', type=str)
    #the path to save the model
    parser.add_argument("--dest_path", default='temp_model.pt', type=str)
    #the path where the finetuned BertRelationNet is, required for training lstm
    parser.add_argument("--path_bert", default='temp_model.pt', type=str)
    parser.add_argument("--learning_rate", default=2e-5, type=np.float32)
    parser.add_argument("--batch_size", default=32, type=np.int32)
    parser.add_argument("--epochs", default=3, type=np.int32)
    parser.add_argument("--weight_decay", default=0, type=np.float32)
    parser.add_argument("--verbose", default=True, type=np.bool8)
    args = parser.parse_args()
    #prepare training and validating data
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    bert = AutoModel.from_pretrained(args.model_name)

    with open(args.path_train_data,'rb') as fb:
        train_data = pickle.load(fb)
    with open(args.path_dev_data,'rb') as fb:
        dev_data = pickle.load(fb)    
    
    relation_list = build_relation_list(args.dataset)
    
    if args.train_option.endswith('bert'):
        if args.train_option.startswith('in_sentence'):
            train_dataloader = prepare_finetune_dataloader(train_data, tokenizer, args.dataset)
            dev_dataloader = prepare_finetune_dataloader(dev_data, tokenizer, args.dataset)
        elif args.train_option.startswith('between_sentence'):
            train_dataloader = prepare_finetune_dataloader(train_data, tokenizer,\
                 args.dataset, between_sentence = True)
            dev_dataloader = prepare_finetune_dataloader(dev_data, tokenizer,\
                 args.dataset, between_sentence = True )
        model = BertRelationNet(bert, relation_list)
    elif args.train_option.endswith('lstm'):
        relation_bert = torch.load(args.path_bert).bert.eval().cuda()
        if args.train_option.startswith('in_sentence'):
            train_dataloader = prepare_seq_label_dataloader(train_data, tokenizer, relation_bert, args.dataset)
            dev_dataloader = prepare_seq_label_dataloader(dev_data, tokenizer, relation_bert, args.dataset)
        elif args.train_option.startswith('between_sentence'):
            train_dataloader = prepare_seq_label_dataloader(train_data, tokenizer,\
                 relation_bert, args.dataset, between_sentence= True)
            dev_dataloader = prepare_seq_label_dataloader(dev_data, tokenizer,\
                 relation_bert, args.dataset, between_sentence= True)
        model = RelationLSTMTagger(relation_list).cuda().double()
    else:
        raise Exception("train_option invalid. Valid train_option: in_sentence_bert, in_sentence_lstm, between_sentence_bert, between_sentence_lstm")


    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    torch.cuda.empty_cache()
    for i in range(args.epochs):
        if args.verbose:
            print(f'for the {i}th epoch')
        if args.train_option.endswith('bert'):
            train_losses.append(classification_train(model, train_dataloader, optimizer, criterion,  verbose = args.verbose))
        else:
            train_losses.append(labeling_train(model, train_dataloader, optimizer, criterion,  verbose = args.verbose))
        with torch.no_grad():
            if args.train_option.endswith('bert'):
                val_losses.append(classification_validate(model, dev_dataloader, criterion,verbose = args.verbose)) 
            else:
                val_losses.append(labeling_validate(model, dev_dataloader, criterion,verbose = args.verbose)) 
    torch.save(model, args.dest_path)

if __name__ == "__main__":
    main()
