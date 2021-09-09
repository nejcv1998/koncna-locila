from transformers import *

# special tokens indices in different models available in transformers
TOKEN_IDX = {
    'bert': {
        'START_SEQ': 101,
        'PAD': 0,
        'END_SEQ': 102,
        'UNK': 100
    },
    'roberta': {
        'START_SEQ': 0,
        'PAD': 1,
        'END_SEQ': 2,
        'UNK': 3
    },
}

# 'O' -> No punctuation
punctuation_dict = {'O': 0, 'EXCLAMATION': 1, 'PERIOD': 2, 'QUESTION': 3}


# pretrained model name: (model class, model tokenizer, output dimension, token style)
MODELS = {
    './cro-slo-eng-bert': (BertModel, BertTokenizer, 768, 'bert'),
    './sloberta': (CamembertModel, CamembertTokenizer, 768, 'roberta'),
}
