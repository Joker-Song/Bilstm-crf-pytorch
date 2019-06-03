class Config(object):

    tagert2idx = {'O':0,'B':1,'I':2,'START':3,'STOP':4} #标签字典

    START = 3
    STOP = 4
    embedding_dim = 256
    hidden_dim = 256
    batch_size = 64
    epochs = 30
    use_gpu = True