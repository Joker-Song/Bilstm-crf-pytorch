import pickle

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from data import NerData
from model import BiLSTM_CRF

from sklearn.metrics import f1_score


def sort_batch_data(sentences, labels, lengths):
    lengths_sort, idx_sort = lengths.sort(0, descending=True)
    sentences_sort = sentences[idx_sort]
    labels_sort = labels[idx_sort]
    _, idx_unsort = idx_sort.sort(0, descending=False)

    return sentences_sort, labels_sort, lengths_sort, idx_unsort


def eval(eval_data, model):
    model.eval()
    dev_dataloader = DataLoader(eval_data,batch_size=32,drop_last=False)

    f1 = 0
    with torch.no_grad():
        for batch_sentence, batch_label, batch_length in dev_dataloader:
            batch_sentence, batch_label, batch_length, _ = sort_batch_data(
                batch_sentence, batch_label, batch_length)
            if Config.use_gpu:
                batch_sentence = batch_sentence.cuda()

            pred = model(batch_sentence, batch_length)
            pred = pred.cpu().numpy()

            true = batch_label[:, :torch.max(batch_length)].numpy()

            for p, t in zip(pred,true):
                f1 += f1_score(t, p, average='macro')

    return f1/len(eval_data)

# train_data 是一个list，list中的每个元素是一个tuple；每个tuple分别包含文本和标签的列表
# 如 [(['我','爱','北','京'],['O','O','B','I']),...]
train_data = pickle.load(open('train_data.pkl', 'rb'))

# 字符到index的字典，其中'<PAD>'对应0
char2idx = pickle.load(open('char2idx.pkl', 'rb'))

dataset = NerData(train_data, char2idx, Config.tagert2idx)

train_dataset, eval_dataset = torch.utils.data.random_split(dataset,(80000,10000))

train_dataloder = DataLoader(
    train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=1,drop_last=False)

model = BiLSTM_CRF(len(char2idx), len(Config.tagert2idx),
                   Config.embedding_dim, Config.hidden_dim)

#默认使用GPU
if Config.use_gpu:
    model = model.to('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

best_score = 0
for epoch in range(Config.epochs):
    model.train()
    total_loss = 0
    for batch_sentence, batch_label, batch_length in train_dataloder:

        model.zero_grad()

        batch_sentence, batch_label, batch_length, _ = sort_batch_data(
            batch_sentence, batch_label, batch_length)
        if Config.use_gpu:
            batch_sentence = batch_sentence.cuda()
            batch_label = batch_label.cuda()
        loss = model.neg_log_likehood(
            batch_sentence, batch_label, batch_length)
        loss.backward()
        optimizer.step()
        total_loss += loss.cpu().item()
    epoch_score = eval(eval_dataset,model)
    if epoch_score > best_score:
        best_score = epoch_score
        torch.save(model.state_dict(), 'model_best.pth')
    print('loss:{0}, epoch_score:{1}, best_score:{2}'.format(total_loss/len(train_dataset),epoch_score,best_score))

