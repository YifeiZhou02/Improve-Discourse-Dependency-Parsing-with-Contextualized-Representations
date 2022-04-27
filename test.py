import sys
sys.path.append('./pytorch_models')
sys.path.append('./utils')
import argparse
from tqdm import tqdm
from UAS_parsing import assembled_sentence_execution, wrapper_model
# import models
from models import BertArcNet, BertRelationNet, RelationLSTMTagger
# from models import BertArctNet, BertRelationNet, RelationLSTMTagger