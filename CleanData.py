import os
import json
import torch
from itertools import islice

from torchtext.data import Field, Example, TabularDataset
from torchtext.data import BucketIterator

def data_prepare(path, file_name, json_name):
    sentence_list = []
    tag_pos_list = []
    tag_syntactic_list = []
    tag_ner_list = []

    sentence = []
    tag_pos = []
    tag_syntactic = []
    tag_ner = []
    
    data_prepare = []
    
    tag_ner_set = set({})
    tag_syntactic_set = set({})
    tag_pos_set = set({})
    with open(os.path.join(path, file_name), "r") as datafile:
        for row in islice(datafile, 2, None):
            tokens = row.strip().split()
            if not len(tokens):
                data_dict = {
                    "sentence": sentence,
                    "tag_pos": tag_pos,
                    "tag_syntactic": tag_syntactic,
                    "tag_ner": tag_ner,
                }
                data_prepare.append(data_dict)
                sentence = []
                tag_pos = []
                tag_syntactic = []
                tag_ner = []
                continue
            sentence.append(tokens[0])
            tag_pos.append(tokens[1])
            tag_syntactic.append(tokens[2])
            tag_ner.append(tokens[3])
            tag_pos_set.add(tokens[1])
            tag_syntactic_set.add(tokens[2])
            tag_ner_set.add(tokens[3])
            
    
    with open(os.path.join(path, json_name) + ".json", 'w') as json_file:
        json_file.write(json.dumps(data_prepare, indent=2))
    with open(os.path.join(path, json_name) + "_tag_pos.json", "w") as pos_file:
        pos_file.write(json.dumps(list(tag_pos_set)))
    with open(os.path.join(path, json_name) + "_tag_syntactic.json", "w") as syntactic_file:
        syntactic_file.write(json.dumps(list(tag_syntactic_set)))
    with open(os.path.join(path, json_name) + "_tag_ner.json", "w") as ner_file:
        ner_file.write(json.dumps(list(tag_ner_set)))

def prepare_data_well():
    data_prepare("data/CoNLL-2003", "eng.train.openNLP", "train")
    data_prepare("data/CoNLL-2003", "eng.testa.openNLP", "testa")
    data_prepare("data/CoNLL-2003", "eng.testb.openNLP", "testb")

def read_data(path):
    with open(path, 'r') as file:
        json_data = json.load(file)
    data_prepare = []
    for data in json_data:
        data_prepare.append((
            data["sentence"],
            data["tag_ner"]
        ))
    return data_prepare

def get_data_large():
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    TAG_SET_CoNLL = ["B-LOC", "I-LOC", "I-PER", "B-MISC", "I-MISC", "B-ORG", "I-ORG", "O",]

    training_data = read_data("data/CoNLL-2003/train.json")
    testa_data = read_data("data/CoNLL-2003/testa.json")
    testb_data = read_data("data/CoNLL-2003/testb.json")

    word_to_ix = {}
    all_data = training_data + testa_data + testb_data
    for sentence, tags in all_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {}
    for index, tag in enumerate(TAG_SET_CoNLL):
        tag_to_ix[tag] = index
    tag_to_ix[START_TAG] = len(TAG_SET_CoNLL)
    tag_to_ix[STOP_TAG] = len(TAG_SET_CoNLL) + 1

    return TAG_SET_CoNLL, tag_to_ix, word_to_ix, training_data, testa_data, testb_data

def get_data_toy():
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    TAG_SET_CoNLL = ["B-LOC", "I-LOC", "I-PER", "B-MISC", "I-MISC", "B-ORG", "I-ORG", "O",]

    training_data = read_data("data/CoNLL-2003/train.json")
    testa_data = read_data("data/CoNLL-2003/testa.json")
    testb_data = read_data("data/CoNLL-2003/testb.json")

    word_to_ix = {}
    all_data = training_data + testa_data + testb_data
    for sentence, tags in all_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {}
    for index, tag in enumerate(TAG_SET_CoNLL):
        tag_to_ix[tag] = index
    tag_to_ix[START_TAG] = len(TAG_SET_CoNLL)
    tag_to_ix[STOP_TAG] = len(TAG_SET_CoNLL) + 1

    return TAG_SET_CoNLL, tag_to_ix, word_to_ix, training_data[:100], testa_data[:30], testb_data[:30]