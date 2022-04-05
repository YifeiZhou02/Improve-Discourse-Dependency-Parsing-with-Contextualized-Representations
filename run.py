from EDU import EDU
from Transition_system import Arc_eager
from models import RelationLSTMTagger, BertArcNet, BertRelationNet
from tqdm import tqdm
from UAS_parsing import assembled_sentence_execution, wrapper_model
import pickle
import torch
import copy
import random
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModel
from relation_labeling import build_relation_list, assembled_transform_heads
# bert = AutoModel.from_pretrained("bert-base-chinese")


def main():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert = AutoModel.from_pretrained("bert-base-uncased")
    with open('preprocessed_data/sci_test.data','rb') as fb:
        test_data = pickle.load(fb)
    in_sentence_model = torch.load('trained_models/SciDTB/sciDTB_in_sentence.pt').eval().cuda()
    between_sentence_model = torch.load('trained_models/SciDTB/sciDTB_between_sentence.pt').eval().cuda()

# with open('cdtb_test.data','rb') as fb:
#   test_data = pickle.load(fb)
# relation_list = build_relation_list('test_data')
# print(relation_list)
# in_sentence_model = torch.load('cdtb_dep/trained_models/cdtb_in_sentence.pt').eval()
# between_sentence_model = torch.load('cdtb_dep/trained_models/cdtb_between_sentence.pt').eval()
    relation_list = build_relation_list('scidtb')

    right = 0
    wrong = 0
    predicted_heads = []
    #do the tree parsing
    for i,edus in tqdm(enumerate(test_data)):
        with torch.no_grad():
            heads = assembled_sentence_execution(edus,wrapper_model(in_sentence_model),\
                                         wrapper_model(between_sentence_model),
                                         tokenizer)
        predicted_heads.append(heads)
        this_right = 0
        this_wrong = 0
        for edu in edus:
            if heads[edu.id] == edu.head:
                right += 1
                this_right += 1
            else:
                wrong += 1
                this_wrong += 1

    in_sentence_model = in_sentence_model.cpu()
    between_sentence_model = between_sentence_model.cpu()

    #load the models for relation tagging
    relation_bert = torch.load('trained_models/SciDTB/sciDTB_relation_bert.pt').bert.eval().cuda()
    relation_bert.gradient_checkpointing_enable()
    between_bert = torch.load('trained_models/SciDTB/sciDTB_between_bert.pt').bert.eval().cuda()
    between_bert.gradient_checkpointing_enable()
    lstm_tagger = torch.load('trained_models/SciDTB/sciDTB_lstm_tagger.pt').eval().cuda().double()
    lstm_tagger.hidden = (lstm_tagger.hidden[0].cuda(),lstm_tagger.hidden[1].cuda())
    between_tagger = torch.load('trained_models/SciDTB/sciDTB_between_tagger.pt').eval().cuda().double()
    between_tagger.hidden = (between_tagger.hidden[0].cuda(), between_tagger.hidden[1].cuda())

    # new_lstm_tagger = RelationLSTMTagger(relation_list)
    # new_lstm_tagger = new_lstm_tagger.double()
    # new_lstm_tagger.lstm = copy.deepcopy(lstm_tagger.lstm)
    # new_lstm_tagger.hidden = copy.deepcopy(lstm_tagger.hidden)
    # lstm_tagger = new_lstm_tagger.eval().cuda()

    #do the relation tagging
    LASright = 0
    LASwrong = 0
    for i,edus in tqdm(enumerate(test_data)):
        heads = predicted_heads[i]
        with torch.no_grad():
            relations = assembled_transform_heads(heads, edus, relation_bert = relation_bert,\
                                lstm_tagger = lstm_tagger, between_relation_bert = between_bert,
                                between_tagger = between_tagger, relation_list = relation_list,
                                tokenizer = tokenizer)
        for edu in edus:
            if heads[edu.id] == edu.head and (relations[edu.id] == edu.relation or edu.relation == 'ROOT'):
                LASright += 1
            else:
                LASwrong+= 1

    print("The number of right is "+ str(right))
    print("The number of wrong is "+ str(wrong))
    print("UAS is "+str(right/(right+wrong)))
    print(f'LAS is {LASright/(LASright + LASwrong)}')

if __name__ == "__main__":
    main()
