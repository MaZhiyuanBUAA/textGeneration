#coding:utf-8
from models import seq2seq
from datautils import tianya

data = tianya()
data.load('../data/data.logits','../data/vocab.json')
data.pad(30)
data.build_trainSet()

model = seq2seq(emb_dim=100,vocab_size=data.vocab_size,encoder_size=data.pad_size,decoder_size=data.pad_size)
model.train(data,128,100,50)