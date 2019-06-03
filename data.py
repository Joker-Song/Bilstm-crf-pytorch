import torch
from torch.utils import data

class NerData(data.Dataset):
    def __init__(self,data,char2idx,tagert2idx):
        self.data = data
        self.max_len = max([len(item[0]) for item in data])
        self.char2idx = char2idx
        self.tagert2idx = tagert2idx
    
    def __getitem__(self,index):
        sentence = self.data[index][0]
        labels = self.data[index][1]
        pad_sentence,pad_labels = self.padding(sentence,labels)
        
        pad_sentence_idx = torch.LongTensor([self.char2idx[i] for i in pad_sentence])
        pad_labels_idx = torch.LongTensor([self.tagert2idx[i] for i in pad_labels])
        length = len(sentence)
        length = torch.tensor(length)
        
        
        return pad_sentence_idx,pad_labels_idx,length
    
    def __len__(self):
        return len(self.data)
    
    def padding(self,sentence,labels):
        pad_sentence = sentence + ['<PAD>']*(self.max_len-len(sentence))
        pad_labels = labels + ['O']*(self.max_len-len(sentence))
        
        return pad_sentence,pad_labels

class PredData(data.Dataset):
    def __init__(self,data,char2idx):
        self.data = data
        self.max_len = max([len(item) for item in data])
        self.char2idx = char2idx
    
    def __getitem__(self,index):
        sentence = self.data[index]
        pad_sentence = self.padding(sentence)
        
        pad_sentence_idx = torch.LongTensor([self.char2idx[i] for i in pad_sentence])
        length = len(sentence)
        length = torch.tensor(length)

        return pad_sentence_idx,length
    
    def __len__(self):
        return len(self.data)
    
    def padding(self,sentence):
        pad_sentence = sentence + ['<PAD>']*(self.max_len-len(sentence))
        
        return pad_sentence
