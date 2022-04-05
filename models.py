#fine-tuning BERT for action classification
#based on the feature of current state
import torch
import torch.nn as nn
from Transition_system import num_EDUs
SEQ_LEN = 80
import pickle
from relation_labeling import build_relation_list

relation_list = build_relation_list('scidtb')

class Bert_net(nn.Module):
    def __init__(self, bert):
        super(Bert_net, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(768*num_EDUs,4)
        self.drop = nn.Dropout(p=0.5)
        self.sf = nn.Softmax(dim = 1)
    def forward(self, x):
        indexes = x.reshape(-1, SEQ_LEN+2)[:,:2]
        x = self.bert(x.reshape(-1, SEQ_LEN+2)[:,2:])[0]
        # print(x)
        features = []
        for i,example in enumerate(x):
          if indexes[i,0] == 0:
            features.append(torch.zeros(1,768).cuda())
          elif indexes[i,0] >= SEQ_LEN:
            features.append(torch.ones(1,768).cuda())
          else:
            features.append(torch.mean(example[indexes[i,0]:indexes[i,1],:], dim = 0).reshape(1,768))
        x = torch.cat(features, dim = 0).reshape(-1, 768*num_EDUs)
        x = self.drop(x)
        x = self.fc1(x)
        return x

#building on BERT for direct relation classification
class Bert_relation_net(nn.Module):
    def __init__(self, bert):
        super(Bert_relation_net, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(768,len(relation_list))
        self.drop = nn.Dropout(p=0.1)
        self.sf = nn.Softmax(dim = 1)
    def forward(self, x):
        x = self.bert(x)[0][:, 0, :]
        x = self.fc1(x)
        return x

#using lstm for relation classification
class Relation_mlp_net(nn.Module):
    def __init__(self, hidden_dim = 768):
        super(Relation_mlp_net, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(768, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.hidden = self.init_hidden()
        self.hidden2action = nn.Linear(768, len(relation_list))
        self.drop = nn.Dropout(p=0.1)
        # self.fc2 = nn.Linear(64,4)
        self.sf = nn.Softmax(dim = 1)
    def init_hidden(self):
        return (torch.zeros(2, 1, self.hidden_dim // 2,dtype = torch.float64),
                torch.zeros(2, 1, self.hidden_dim // 2, dtype = torch.float64))
    def forward(self, features):
        features , _ = self.lstm(features.view(-1, 1, 768), self.hidden)
        x = self.drop(features)
        x = self.hidden2action(x)
        return x.view(-1,len(relation_list))

class BertArcNet(nn.Module):
    def __init__(self, bert):
        super(BertArcNet, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(768*num_EDUs,4)
        self.drop = nn.Dropout(p=0.5)
        self.sf = nn.Softmax(dim = 1)
    def forward(self, x):
        indexes = x.reshape(-1, SEQ_LEN+2)[:,:2]
        x = self.bert(x.reshape(-1, SEQ_LEN+2)[:,2:])[0]
        # print(x)
        features = []
        for i,example in enumerate(x):
          if indexes[i,0] == 0:
            features.append(torch.zeros(1,768).cuda())
          elif indexes[i,0] >= SEQ_LEN:
            features.append(torch.ones(1,768).cuda())
          else:
            features.append(torch.mean(example[indexes[i,0]:indexes[i,1],:], dim = 0).reshape(1,768))
        x = torch.cat(features, dim = 0).reshape(-1, 768*num_EDUs)
        x = self.drop(x)
        x = self.fc1(x)
        return x

#building on BERT for direct relation classification
class BertRelationNet(nn.Module):
    def __init__(self, bert, relation_list):
        super(BertRelationNet, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(768,len(relation_list))
        self.drop = nn.Dropout(p=0.1)
        self.sf = nn.Softmax(dim = 1)
    def forward(self, x):
        x = self.bert(x)[0][:, 0, :]
        x = self.fc1(x)
        return x

#using lstm for relation classification
class RelationLSTMTagger(nn.Module):
    def __init__(self, relation_list, hidden_dim = 768):
        super(RelationLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(768, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.hidden = self.init_hidden()
        self.hidden2action = nn.Linear(768, len(relation_list))
        self.drop = nn.Dropout(p=0.1)
        self.relation_list = relation_list
        # self.fc2 = nn.Linear(64,4)
        self.sf = nn.Softmax(dim = 1)
    def init_hidden(self):
        return (torch.zeros(2, 1, self.hidden_dim // 2,dtype = torch.float64),
                torch.zeros(2, 1, self.hidden_dim // 2, dtype = torch.float64))
    def forward(self, features):
        features , _ = self.lstm(features.view(-1, 1, 768), self.hidden)
        x = self.drop(features)
        x = self.hidden2action(x)
        return x.view(-1,len(self.relation_list))