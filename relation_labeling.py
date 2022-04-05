import numpy as np
import torch


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
                              return_attention_mask = False, return_tensors = 'tf')
      Xids = tokens['input_ids'].numpy()
      # if it is the second component discard the CLS
      if i == 1:
        Xids = Xids[:,1:]
      new_f.append(torch.Tensor(Xids))
    new_features.append(torch.cat(new_f, dim = -1).reshape(1,-1))
  return  torch.cat(new_features, dim = 0).long()

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