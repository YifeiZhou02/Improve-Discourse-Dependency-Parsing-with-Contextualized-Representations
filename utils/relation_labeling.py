import numpy as np
import torch
from utils.training_utils import build_between_sentence_data

def build_relation_list(option = 'scidtb'):
  """
  iterate over data to yield a list of all relations
  """
  if option == 'scidtb':
    return ['ROOT',
    'elab-aspect',
    'elab-addition',
    'enablement',
    'same-unit',
    'contrast',
    'attribution',
    'evaluation',
    'bg-goal',
    'manner-means',
    'elab-enum_member',
    'joint',
    'null',
    'elab-definition',
    'bg-compare',
    'elab-example',
    'cause',
    'result',
    'progression',
    'temporal',
    'bg-general',
    'condition',
    'exp-reason',
    'summary',
    'comparison',
    'exp-evidence',
    'elab-process_step']
  return ['ROOT',
    '因果关系',
    '背景关系',
    '转折关系',
    '并列关系',
    '目的关系',
    '例证关系',
    '解说关系',
    '条件关系',
    '总分关系',
    '假设关系',
    '顺承关系',
    '对比关系',
    '递进关系',
    '评价关系',
    '推断关系',
    '让步关系',
    '选择关系']

#tokenize the relation data 
#input a list of feature pairs
def tokenize_relation_data(features, tokenizer, SEQ_LEN = 40):
  """
  input a list of tuples of EDUs as relation pair feature
  output the concatenated tokenized version
  """
  new_features = []
  for f in features:
    new_f = []
    for i,edu in enumerate(f):
      if edu == None:
        sentence = ''
      else:
        sentence = edu.sentence
      tokens = tokenizer.encode_plus(sentence, max_length = SEQ_LEN,
                              truncation = True, padding = "max_length",
                              add_special_tokens = True, return_token_type_ids = False,
                              return_attention_mask = False, return_tensors = 'pt')
      Xids = tokens['input_ids'].numpy()
      # if it is the second component discard the CLS
      if i == 1:
        Xids = Xids[:,1:]
      new_f.append(torch.Tensor(Xids))
    new_features.append(torch.cat(new_f, dim = -1).reshape(1,-1))
  return  torch.cat(new_features, dim = 0).long()

def transform_heads_simple(heads, edus, simple_relation_net, relation_list, tokenizer):
  """
  using the baseline bert to predict relations
  return the predicted relations by direct relation classification
  heads is a dictionary where heads[edu.id] is the id of its head
  """
  simple_relation_net = simple_relation_net.cuda()
  relations = {}
  n = len(heads)
  for edu in range(1,1+n):
    #edu is the id of this edu
    #if it is root, label it as ROOT relation
    head = heads[edu]
    if head == 0:
      feature = [[None, edus[edu -1]]]
    else:
      if head < edu:
        feature = [[edus[head - 1], edus[edu - 1]]]
      else:
        feature = [[ edus[edu - 1], edus[head - 1]]]
    feature = tokenize_relation_data(feature, tokenizer)
    relation = np.argmax(simple_relation_net(feature.cuda()).cpu().detach().numpy())
    relations[edu] = relation_list[relation]
  return relations

#assemble post processing to transform heads
def assembled_transform_heads(heads, edus, relation_bert, lstm_tagger, between_relation_bert, 
                              between_tagger, relation_list, tokenizer):
    """
    predict relations from respect dependency structure
    correspond to the BERT + Stacked BiLSTM model
    return a dictionary of relations
    edus start from id 0
    inputs a relation bert as encoder and an lstm tagger as decoder
    Args:
        - :param: `heads` (list of integers): the id of the head for each edu
        - :param: `edus` (list of EDU): a list of edus
        - :param: `relation_bert` (transformer model): the transformer model used to get
            the embeddings for first-level sequence labeling
        - :param: `lstm_tagger` (sequence labeling model): the sequence labeling model 
            used to get the embeddings for first-level sequence labeling
        - :param: `between_relation_bert` (transformer model): the transformer model used 
            to get the embeddings for second-level sequence labeling
        - :param: `between_tagger` (sequence labeling model): the sequence labeling model 
            used to get the embeddings for second-level sequence labeling
        - :param: `relation_list` (list of str): a list of all relations
    Return:
        - :param: 'relations' (disctionary): each (key, value) pair in the dictionary, 
            key is the id of the edu, value is 
    """
    relations = {}
    relation_features = []
    #the position encodings
    position = np.array([i for i in range(768)])
    position_enc = np.power(10000,position/768)
    n = len(heads)
    between_heads = {}
    between_edus = []
    #between features is the feature for post tagging between sentences
    between_features = []
    for edu in range(1,1+n):
      #edu is the id of this edu
      #if it is root, label it as ROOT relation
      head = heads[edu]
      if head == 0:
        feature = [[None, edus[edu -1]]]
      else:
        if head < edu:
          feature = [[edus[head - 1], edus[edu - 1]]]
        else:
          feature = [[edus[edu -1], edus[head - 1]]]
      feature = tokenize_relation_data(feature, tokenizer)
      #append feature to between sentence
      sin_enc = np.sin(edus[edu -1].id/position_enc)+np.cos(edus[edu-1].sentenceID/position_enc)
      if head == 0 or edus[edu - 1].sentenceNo != edus[head - 1].sentenceNo:
        between_heads[edu] = head
        between_edus.append(edu)
        between_features.append(between_relation_bert(feature.cuda())[0][:,0,:].cpu().detach() + sin_enc)

      feature = relation_bert(feature.cuda())[0][:,0,:].cpu().detach() + sin_enc
      relation_features.append(feature)
    relation_features = torch.cat(relation_features, dim = 0)
    all_relations = np.argmax(lstm_tagger(relation_features.cuda()).cpu().detach().numpy(), axis = 1)
    for i, relation_index in enumerate(all_relations):
      relations[i+1] = relation_list[relation_index]
    between_features = torch.cat(between_features, dim = 0)
    between_relations = np.argmax(between_tagger(between_features.cuda()).cpu().detach().numpy(), axis = 1)
    for i, relation_index in enumerate(between_relations):
      relations[between_edus[i]] = relation_list[relation_index]
    return relations

def build_paired_data(data, option = 'scidtb', between_sentence = False):
  """
  build features and labels for direct relation classification
  each feature is a tuple of two edus (in the same order as they appear 
  in the discourse
  option can be scidtb or cdtb
  """
  relation_list = build_relation_list(option)
  between_sentence_data = build_between_sentence_data(data)
  features = []
  labels = []
  for edus, between_edus in zip(data, between_sentence_data):
    for edu in edus:
      if between_sentence and (not edu in between_edus):
        continue
      if edu.head == 0:
        features.append([None, edu])
      else:
        if edu.id > edu.head:
          features.append([edus[edu.head - 1], edu])
        else:
          features.append([edu, edus[edu.head - 1]])
      labels.append(relation_list.index(edu.relation))
  return features, torch.Tensor(labels)

def prepare_finetune_dataloader(data, tokenizer, option = 'scidtb',\
   SEQ_LEN = 40, between_sentence = False):
  """
  prepare the dataloader for finetuning
  """
  features, labels = build_paired_data(data, option, between_sentence)
  features = tokenize_relation_data(features, tokenizer, SEQ_LEN)
  X_train = features.long()
  y_train = labels.long()
  train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last = True)
  return train_dataloader

#a dataset structure for sequence labeling
class LSTMDataset(torch.utils.data.Dataset):
    def __init__(self, features,labels):
      self.features = features
      self.labels = labels
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
       return [self.features[idx], self.labels[idx].flatten().long()]

def prepare_seq_label_dataloader(data, tokenizer, relation_bert ,option = 'scidtb', SEQ_LEN = 40,\
  between_sentence = False):
  """
  prepare the dataloader to train sequence labeling models
  """
  relation_bert = relation_bert.cuda()
  edu2tokens = {}
  relation_list = build_relation_list(option)
  for edus in data:
    for edu in edus:
      if edu.head == 0:
        features = [[None, edu]]
      else:
        if edu.head < edu.id:
          features = [[edus[edu.head - 1], edu]]
        else:
          features = [[edu, edus[edu.head - 1]]]
      edu2tokens[edu]= tokenize_relation_data(features, tokenizer, SEQ_LEN).long().reshape(1,-1)
  

  #turn pair tokens to bert representations
  edu2representations = {}
  if between_sentence:
    data = build_between_sentence_data(data)
  for j,edus in enumerate(data):
    token_features = torch.cat([edu2tokens[edu] for edu in edus], dim = 0)
    token_features = relation_bert(token_features.cuda())[0][:,0,:].detach().cpu()
    for i,edu in enumerate(edus):
      edu2representations[edu] = token_features[i]
  
    #add position encodings
  position = np.array([i for i in range(len(edu2representations[data[0][0]]))])
  position_enc = np.power(10000,position/768)
  for edus in data:
    for edu in edus:
      sin_enc = np.sin(edu.id/position_enc)+np.cos(edu.sentenceID/position_enc)
      edu2representations[edu] = edu2representations[edu] + sin_enc
  #use ed2representations to build sequence tagging of lstm
  features = []
  labels = []
  for edus in data:
    features.append(torch.cat([edu2representations[edu].reshape(1,-1) for edu in edus], dim = 0))
    labels.append(torch.Tensor([relation_list.index(edu.relation)for edu in edus]).long() )

  train_dataloader = LSTMDataset(features, labels)
  return train_dataloader

#training for sequence labeling
def labeling_train(net, train_dataloader, optimizer, criterion, verbose = False):
    net = net.train().cuda()
    net.hidden = (net.hidden[0].cuda(), net.hidden[1].cuda())
    losses = []
    weights = []
    i = 0
    for batch_idx, (inputs, target) in enumerate(train_dataloader):
        outputs = net(inputs.cuda()).cpu()
        loss = criterion(outputs, target)
        loss.backward()
        if i == 32:
          optimizer.step()
          optimizer.zero_grad()
          i = 0
        losses.append(loss.item())
        weights.append(len(target))
        i += 1
    losses = np.array(losses)
    weights = np.array(weights)
    result_loss = np.sum(losses*weights)/np.sum(weights)
    if verbose:
      print(f'the training loss is {result_loss}')
    return result_loss
    
#validate the model, return the loss
def labeling_validate(net, val_dataloader, criterion, verbose = False):
    net = net.eval().cuda()
    net.hidden = (net.hidden[0].cuda(), net.hidden[1].cuda())
    losses = []
    weights = []
    accuracies = []
    for batch_num, (inputs, target) in enumerate(val_dataloader):
        outputs = net(inputs.cuda()).cpu()
        accuracy = torch.sum(torch.argmax(outputs, dim = 1) == target)
        accuracies.append(accuracy)
        loss = criterion(outputs, target)
        losses.append(loss.item())
        weights.append(len(target))
    losses = np.array(losses)
    weights = np.array(weights)
    accuracies = np.array(accuracies)
    result_loss = np.sum(losses*weights)/np.sum(weights)
    if verbose:
      print(f'the validation loss is {result_loss}')
      print(f'the validation accuracy is {np.sum(accuracies)/np.sum(weights)}')
    return result_loss
