import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import json

from BiLSTM_CRF import BiLSTM_CRF
from CleanData import get_data_large, get_data_toy

import time

gpu_available = torch.cuda.is_available()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

TAG_SET_CoNLL, tag_to_ix, word_to_ix, training_data, testa_data, testb_data = get_data_large()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_loaded = torch.load("saved_models/model_ver_5.pkl", map_location='cuda')
model_loaded.eval()
# toy_training_data = [training_data[0]]

training_data = training_data + testa_data

with torch.no_grad():
    recall_train_denominator = 0
    precision_train_denominator = 0
    acc_train_numerator = 0
    # for sentence, tags in training_data:
    for sentence, tags in training_data:
        sentence_in = prepare_sequence(sentence, word_to_ix)
        if len(sentence_in) < 2:
            continue
        if gpu_available:
            sentence_in = sentence_in.cuda()
        real_tagsets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).tolist()
        predicted_tagsets = model_loaded(sentence_in)[1]
        # print(real_tagsets)
        # print(predicted_tagsets)
        for real_tag, predicted_tag in zip(real_tagsets, predicted_tagsets):
            if int(real_tag) != 7:
                recall_train_denominator += 1
                if int(real_tag) == int(predicted_tag):
                    acc_train_numerator += 1
            if int(predicted_tag) != 7:
                precision_train_denominator += 1
    recall = acc_train_numerator / recall_train_denominator
    precision = acc_train_numerator / precision_train_denominator
    F1_score = 2 * recall * precision / (recall + precision)

    print("training data:")
    print(recall)
    print(precision)
    print(F1_score)
    
    recall_dev_denominator = 0
    precision_dev_denominator = 0
    acc_dev_numerator = 0
    # for sentence, tags in deving_data:
    for sentence, tags in testb_data:
        sentence_in = prepare_sequence(sentence, word_to_ix)
        if len(sentence_in) < 2:
            continue
        if gpu_available:
            sentence_in = sentence_in.cuda()
        real_tagsets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).tolist()
        predicted_tagsets = model_loaded(sentence_in)[1]
        # print(real_tagsets)
        # print(predicted_tagsets)
        for real_tag, predicted_tag in zip(real_tagsets, predicted_tagsets):
            if int(real_tag) != 7:
                recall_dev_denominator += 1
                if int(real_tag) == int(predicted_tag):
                    acc_dev_numerator += 1
            if int(predicted_tag) != 7:
                precision_dev_denominator += 1
    recall = acc_dev_numerator / recall_dev_denominator
    precision = acc_dev_numerator / precision_dev_denominator
    F1_score = 2 * recall * precision / (recall + precision)

    print("dev data:")
    print(recall)
    print(precision)
    print(F1_score)

