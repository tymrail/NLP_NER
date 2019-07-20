import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.nn import functional as F

import numpy as np
import pandas as pd
import os
import time
import gc
import random
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from keras.preprocessing import text, sequence

import json

from BiLSTM_CRF import BiLSTM_CRF
from CleanData import get_data_large, get_data_toy

import time

torch.manual_seed(1)

gpu_available = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GLOVE_EMBEDDING_PATH = 'glove/glove.6B.200d.txt'

if gpu_available:
    torch.cuda.manual_seed(1)

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path, 'r', encoding="utf-8") as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))

def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 200))
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

now = int(time.time())
file_name = str(now) + "_run.txt"
with open("runs/" + file_name, "w+") as f:
    dt = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)))
    f.write("Start training at " + dt + "\n")

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 200
HIDDEN_DIM = 200

TAG_SET_CoNLL, tag_to_ix, word_to_ix, training_data, testa_data, testb_data = get_data_large()
# TAG_SET_CoNLL, tag_to_ix, word_to_ix, training_data, testa_data, testb_data = get_data_toy()

# print(training_data)

training_data = training_data + testa_data

glove_matrix, unknown_words = build_matrix(word_to_ix, GLOVE_EMBEDDING_PATH)
print('n unknown words (glove):', len(unknown_words))

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, embedding_matrix=glove_matrix)
model.train()

if gpu_available:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    if gpu_available:
        precheck_sent = precheck_sent.cuda()
        precheck_tags = precheck_tags.cuda()
    print(model(precheck_sent))

for epoch in range(48):
    global_loss = 0
    index_t = 0
    for sentence, tags in training_data:
        model.zero_grad()

        sentence_in = prepare_sequence(sentence, word_to_ix)

        if len(sentence_in) < 2:
            continue

        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        if gpu_available:
            sentence_in = sentence_in.cuda()
            targets = targets.cuda()

        loss = model.neg_log_likelihood(sentence_in, targets)

        if gpu_available:
            loss = loss.cuda()

        loss.backward()
        optimizer.step()

        global_loss += loss.item()

        index_t += 1
        if index_t % 5000 == 0:
            # print("Epoch:" + str(epoch) + ", progess: " + str(index_t / len(training_data) * 100) + "%, avg_loss:" + str(global_loss / index_t))
            with open("runs/" + file_name, "a+") as f:
                now_t = int(time.time())
                dt_t = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_t)))
                f.write("|Epoch:" + str(epoch) + "|progess: " + str(index_t / len(training_data) * 100) + "%|avg_loss:" + str(global_loss / index_t) + "|now:" + str(dt_t) + "\n")
    
    with open("runs/" + file_name, "a+") as f:
        f.write("|Epoch:" + str(epoch) + "|avg_loss:" + str(global_loss / index_t) + "|\n")
    print("Epoch:" + str(epoch) + "|avg_loss:" + str(global_loss / len(training_data)) + "|")

torch.save(model, "saved_models/model_ver_5.pkl")

print("Training Done.")

with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))

# model_loaded = torch.load("saved_models/toy_model_2.pkl")
# toy_training_data = [training_data[0]]

# with torch.no_grad():
#     recall_train_denominator = 0
#     precision_train_denominator = 0
#     acc_train_numerator = 0
#     # for sentence, tags in training_data:
#     for sentence, tags in toy_training_data:
#         sentence_in = prepare_sequence(sentence, word_to_ix)
#         real_tagsets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).tolist()
#         predicted_tagsets = model_loaded(sentence_in)[1]
#         print(real_tagsets)
#         print(predicted_tagsets)
#         for real_tag, predicted_tag in zip(real_tagsets, predicted_tagsets):
#             if int(real_tag) != 7:
#                 recall_train_denominator += 1
#                 if int(real_tag) == int(predicted_tag):
#                     acc_train_numerator += 1
#             if int(predicted_tag) != 7:
#                 precision_train_denominator += 1
#     recall = acc_train_numerator / recall_train_denominator
#     precision = acc_train_numerator / precision_train_denominator
#     F1_score = 2 * recall * precision / (recall + precision)

#     print(recall)
#     print(precision)
#     print(F1_score)
    
            

