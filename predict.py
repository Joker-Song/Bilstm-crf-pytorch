import pickle

import torch
from torch.utils.data import DataLoader

from config import Config
from data import PredData
from model import BiLSTM_CRF

def sort_batch_data(sentences, lengths):
    lengths_sort, idx_sort = lengths.sort(0, descending=True)
    sentences_sort = sentences[idx_sort]
    _, idx_unsort = idx_sort.sort(0, descending=False)

    return sentences_sort, lengths_sort, idx_unsort

char2idx = pickle.load(open('char2idx.pkl', 'rb'))
data = pickle.load(open('predict_data.pkl','rb'))

predict_data = PredData(data,char2idx)
dataloader = DataLoader(predict_data,batch_size=32,drop_last=False)

model = BiLSTM_CRF(len(char2idx), len(Config.tagert2idx),
                   Config.embedding_dim, Config.hidden_dim)

model.load_state_dict(torch.load('model_best.pth'))
if Config.use_gpu:
    model.to('cuda')
model.eval()

predict_result = []
with torch.no_grad():
    for batch_sentences,batch_lengths in dataloader:
        sentences,lengths,idx_unsort = sort_batch_data(batch_sentences,batch_lengths)
        if Config.use_gpu:
            sentences = sentences.cuda()
        pred = model(sentences,lengths)
        pred = pred[idx_unsort]
        pred = pred.cpu().numpy()

        ls = batch_lengths.numpy()

        for tags,l in zip(pred,ls):
            predict_result.append(list(tags)[:l])

pickle.dump(predict_result,open('predict_result5.pkl','wb'))
print(predict_result[33])
print('Done')