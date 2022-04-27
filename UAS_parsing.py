from EDU import EDU
from Transition_system import Arc_eager, num_EDUs
import torch

def modify_contextualized_embeddings(edus, tokenizer, SEQ_LEN = 80):
  """
  update contextualized embeddings for edus
  no_context: if true, use the embeddings itself without the context
  """
      
  sentence = ''
  for e in edus:
    sentence += e.sentence
  sentence_tokens = tokenizer.encode_plus(sentence, max_length = SEQ_LEN,
                              truncation = True, padding = "max_length",
                              add_special_tokens = True, return_token_type_ids = False,
                              return_attention_mask = False, return_tensors = 'pt')['input_ids'].numpy()
  edu2length = {}
  for edu in edus:
    tokens = tokenizer.encode_plus(edu.sentence,
                              add_special_tokens = False, return_token_type_ids = False,
                              return_attention_mask = False, return_tensors = 'pt')
    Xids = tokens['input_ids'].numpy()
    edu2length[edu] = len(Xids.flatten())
  for edu in edus:
    start = 1
    for e in edus:
      if e == edu:
        break
      start += edu2length[e]
    end = start + edu2length[edu]
    edu.embeddings = torch.Tensor([start, end]).long()
    edu.embeddings = torch.cat([edu.embeddings, torch.Tensor(sentence_tokens).flatten()], dim = -1).long()
  return edus

def wrapper_model( model):
  """
  a model wrapper to deal with some overhead work
  """
  model = model.cuda()
  def my_wrapper_model(x):
    x = torch.Tensor(x).long().cuda()
    result = model(x).cpu().detach().numpy()
    return result
  return lambda x:my_wrapper_model(x)

def assembled_sentence_execution(edus,in_sentence_model, between_sentence_model, tokenizer):
    """
    use both the in_sentence_model (intra-sententential tree constructor) and
    the between_sentence_model (inter-sentential tree constructor)
    to assemble a complete discourse tree in a Sent-First manner
    Note that the embeddings of edus might be changed after execution
    """
    in_sentence_data = []
    in_sentence_data.append([])
    n = len(edus)
    for j,edu in enumerate(edus):
        in_sentence_data[-1].append(edu)
        if j < n -1 and edus[j].sentenceNo != edus[j+1].sentenceNo:
            in_sentence_data.append([])
    all_heads = {}
    for sentence_edus in in_sentence_data:
        sentence_edus = modify_contextualized_embeddings(sentence_edus, 
                                                         tokenizer)
        trans = Arc_eager(sentence_edus)
        heads = trans.model_execute(in_sentence_model)
        for key, value in heads.items():
            all_heads[key] = value
    between_sentence_data = []
    for key, value in all_heads.items():
        if value == 0:
            between_sentence_data.append(edus[key - 1])
    between_sentence_data = modify_contextualized_embeddings(between_sentence_data,
                                                             tokenizer)
    trans = Arc_eager(between_sentence_data)
    heads = trans.model_execute(between_sentence_model)
    for key, value in heads.items():
        all_heads[key] = value
    return all_heads