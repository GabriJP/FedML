import collections

import numpy as np

_pad = '<pad>'
_bos = '<bos>'
_eos = '<eos>'
'''
This code follows the steps of preprocessing in tff shakespeare dataset: 
https://github.com/google-research/federated/blob/master/utils/datasets/shakespeare_dataset.py
'''

SEQUENCE_LENGTH = 80  # from McMahan et al AISTATS 2017
# Vocabulary re-used from the Federated Learning for Text Generation tutorial.
# https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
CHAR_VOCAB = list(
    'dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r'
)

word_dict = collections.OrderedDict((w, i) for i, w in enumerate([_pad] + CHAR_VOCAB + [_bos] + [_eos]))
word_list = tuple(word_dict.keys())


def char_to_id(char):
    return word_dict.get(char, len(word_dict))


def preprocess(sentences, max_seq_len=SEQUENCE_LENGTH):
    sequences = list()

    def to_ids(sentence):
        """
        map list of sentence to list of [idx..] and pad to max_seq_len + 1
        Args:
            sentence: String to be processed
        """
        tokens = list(map(char_to_id, sentence))
        tokens = [char_to_id(_bos)] + tokens + [char_to_id(_eos)]
        if len(tokens) % (max_seq_len + 1) != 0:
            pad_length = (-len(tokens)) % (max_seq_len + 1)
            tokens += [char_to_id(_pad)] * pad_length
        return (tokens[i:i + max_seq_len + 1] for i in range(0, len(tokens), max_seq_len + 1))

    for sen in sentences:
        sequences.extend(to_ids(sen))
    return sequences


def split(dataset):
    ds = np.asarray(dataset)
    x = ds[:, :-1]
    y = ds[:, 1:]
    return x, y


if __name__ == "__main__":
    print(split(preprocess([
        'Yonder comes my master, your brother.',
        'Come not within these doors; within this roof\nThe enemy of all your graces lives.\nYour brother- no, no '
        'brother; yet the son-\nYet not the son; I will not call him son\nOf him I was about to call his father-\nHath '
        'heard your praises; and this night he means\nTo burn the lodging where you use to lie,\nAnd you within it. If '
        'he fail of that,\nHe will have other means to cut you off;\nI overheard him and his practices.\nThis is no '
        'place; this house is but a butchery;\nAbhor it, fear it, do not enter it.\nNo matter whither, so you come '
        'not here.',
        'To the last gasp, with truth and loyalty.\nFrom seventeen years till now almost four-score\nHere lived I, but '
        'now live here no more.\nAt seventeen years many their fortunes seek,\nBut at fourscore it is too late a week;'
        '\nYet fortune cannot recompense me better\nThan to die well and not my master\'s debtor.          Exeunt\n'
        'Dear master, I can go no further. O, I die for food! Here lie',
        '[Coming forward] Sweet masters, be patient; for your father\'s',
        "remembrance, be at accord.\nIs 'old dog' my reward? Most true, I have lost my teeth in",
    ])))
