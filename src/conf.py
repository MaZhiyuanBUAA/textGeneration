#coding:utf-8
#from text to logits
class txt2logit_conf(object):
    def __init__(self):
        self.UNK = 0
        self.START = 1
        self.END = 2
        self.PAD = 3
    # def add_path(self,jsonDataPath,txtDataPath,logitsDataPath,vocabPath,posDataPath,posDictPath):
    #     self.jsonDataPath = jsonDataPath
    #     self.txtDataPath = txtDataPath
    #     self.logitsDataPath = logitsDataPath
    #     self.vocabPath = vocabPath
    #     self.posDataPath = posDataPath
    #     self.posDictPath = posDictPath