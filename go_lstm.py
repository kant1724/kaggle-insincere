import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from torch.nn.utils import clip_grad_norm_
import pandas as pd
from torch.autograd import Variable
import math
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.stem.lancaster import LancasterStemmer
lc = LancasterStemmer()
from nltk.stem import SnowballStemmer
sb = SnowballStemmer("english")
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Hyper parameters
num_epochs = 15
num_classes = 1
batch_size = 512
learning_rate = 0.001
embedding_dim = 2 * 300
hidden_size = 128
num_layers = 1
threshold = 0.35
max_features = 50000
maxlen = 100

train_df = pd.read_csv('./train.csv')

test_df = pd.read_csv('./test.csv')
test2_df = pd.read_csv('./test.csv')

test_input = np.array(test_df['question_text'].values)
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_df['question_text'].values))
train_X = tokenizer.texts_to_sequences(train_df['question_text'].values)
test_X = tokenizer.texts_to_sequences(test_df['question_text'].values)

## Pad the sentences
train_X = pad_sequences(train_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)
train_Y = train_df['target'].values

def load_glove(word_dict):
    EMBEDDING_FILE = './embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf8'))
    embed_size = 300
    nb_words = len(word_dict)+1
    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)
    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.
    print(unknown_vector[:5])
    for key in word_dict:
        word = key
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = ps.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lc.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = sb.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        embedding_matrix[word_dict[key]] = unknown_vector
    return embedding_matrix, nb_words

def load_fasttext(word_dict):
    EMBEDDING_FILE = './embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf8') if len(o)>100)
    embed_size = 300
    nb_words = len(word_dict)+1
    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)
    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.
    print(unknown_vector[:5])
    for key in word_dict:
        word = key
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = ps.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lc.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = sb.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        embedding_matrix[word_dict[key]] = unknown_vector
    return embedding_matrix, nb_words

class GlobalMaxPooling1D(nn.Module):
    def __init__(self):
        super(GlobalMaxPooling1D, self).__init__()

    def forward(self, inputs):
        z, _ = torch.max(inputs, 1)
        return z

    def __repr__(self):
        return self.__class__.__name__ + '()'
# RNN based language model

class RNNLM(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, pretrained_embedding):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embedding))
        print(np.shape(self.embed))
        self.proj = nn.Linear(embed_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size * 2, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 4, hidden_size)
        self.pooling = GlobalMaxPooling1D()
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embed(x)
        out = torch.relu(self.proj(x))
        out, (h, c) = self.gru(out)
        out, (h, c) = self.lstm(out)
        out = self.pooling(out)
        out = torch.relu(self.linear(out))
        out = self.dense(out)

        return out, (h, c)

    def predict(self, x):
        preds = []
        with torch.no_grad():
            x, _ = self.forward(x)
            preds.append(x)
        return torch.cat(preds)

    def predict_proba(self, x):
        return torch.sigmoid(self.predict(x))

embedding_matrix_glove, _ = load_glove(tokenizer.word_index)
embedding_matrix_fasttext, _ = load_fasttext(tokenizer.word_index)

embedding_matrix = np.concatenate((embedding_matrix_glove, embedding_matrix_fasttext), axis=1)

model = RNNLM(embedding_dim, hidden_size, num_layers, embedding_matrix).cuda()

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def detach(states):
    return [state.detach() for state in states]

train_input = torch.LongTensor(train_X)
train_target = torch.FloatTensor(train_Y)
tot = int(len(train_input) / batch_size)
for epoch in range(num_epochs):
    for i in range(tot):
        train_input = train_input.cuda()
        train_target = train_target.cuda()
        fr = min(i * batch_size, len(train_input))
        to = min((i + 1) * batch_size, len(train_input))
        outputs, states = model(train_input[fr:to])
        loss = criterion(np.squeeze(outputs), train_target[fr:to])

        model.zero_grad()
        loss.backward()
        optimizer.step()

test_input = torch.LongTensor(test_X)
tot = int(len(test_input) / batch_size)
out = []
with torch.no_grad():
    test_input = test_input.cuda()
    for i in range(tot):
        fr = min(i * batch_size, len(test_input))
        to = min((i + 1) * batch_size, len(test_input))
        res = model.predict_proba(test_input[fr:to])
        res = res.cpu().numpy()
        for r in res:
            out.append(r)

result = []
out = np.squeeze(out)
print(out)
for o in out:
    if o > threshold:
        result.append(1)
    else:
        result.append(0)

total_len = len(test2_df.qid)
while len(result) < total_len:
    result.append(0)

output = pd.DataFrame({'qid': test2_df.qid, 'prediction': result})
output.to_csv('submission.csv', index=False)
