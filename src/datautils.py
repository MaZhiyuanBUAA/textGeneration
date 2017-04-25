#coding:utf-8
import json,os,random,pickle
import numpy as np
import jieba
from conf import txt2logit_conf
import jieba.posseg as pseg

def basic_filter(p,c,keysWords=[]):
    if (not p) or (not c):
        return 0
    else:
        tmp = p + c
        for word in keysWords:
            if tmp.find(word)>-1:
                return 0
    return 1

class data_set(txt2logit_conf):
    def __init__(self,name='data_set'):
        txt2logit_conf.__init__(self)
        self.name = name
    def load_data(self, path, vocab_path):
        print('load from %s' % path)
        f = open(path)
        data_txt = f.readlines()
        self.data_size = len(data_txt)
        f.close()
        self.data_logits = [[int(k) for k in ele.split(' ')] for ele in data_txt]
        f = open(vocab_path)
        self.vocab = json.loads(f.read())
        self.vocab_size = len(self.vocab)
        f.close()
        print('Info:\ndata_size:%d\nvocab_size:%d' % (self.data_size, self.vocab_size))
        doc_table = np.ones(self.vocab_size)
        for ele in self.data_logits:
            tmp = np.zeros(self.vocab_size)
            for k in ele:
                tmp[k] = 1
            doc_table += tmp
        self.idf_table = np.log(np.sum(doc_table) / doc_table)
        print('IDF table successfully getted!')
    def load_POS(self,pos_path,pos_dict):
        print('load from %s' % pos_path)
        f = open(pos_path)
        pos_txt = f.readlines()
        f.close()
        self.pos_logits = [[int(k) for k in ele.split(' ')] for ele in pos_txt]
        f = open(pos_dict)
        self.pos_vocab = json.loads(f.read())
        f.close()
        self.pos_size = len(self.pos_vocab)
        print('POS info successfully loaded!')
    def sent2logits(self,sentence,pad_size=False):
        sentence = sentence.replace('\n','')
        words = jieba.cut(sentence)
        tmp = []
        for w in words:
            try:
                tmp.append(self.vocab[w])
            except:
                tmp.append(self.UNK)
        if pad_size:
            tmp = [self.START] + tmp + [self.END]
            l = len(tmp)
            if l <= pad_size:
                return tmp+(pad_size-l)*[self.PAD]
            else:
                return tmp[:pad_size]
        else:
            return tmp
    def tf_idf(self,sentence):
        logits = self.sent2logits(sentence)
        s_size = len(logits)
        if s_size == 0:
            print('Bad sentence (>_<)')
            return
        tmp = np.zeros(self.vocab_size)
        for k in logits:
            tmp[k] += 1
        return 1.*tmp/s_size*self.idf_table

    def pad(self,pad_size):
        self.pad_size = pad_size
        pad_data = []
        for ele in self.data_logits:
            l = len(ele)
            if l >= pad_size:
                pad_data.append(ele[:pad_size])
            else:
                pad_data.append(ele+(pad_size-l)*[self.PAD])
        self.pad_data = pad_data

    def build_trainSet(self, rate=(0.7, 0.9), seed=1):
        random.seed(seed)
        try:
            self.data_size += 0
        except:
            print('No data !!!!')
            return
        nums = self.data_size * np.array(rate) / 2
        nums = nums.astype(int)
        inds = list(range(int(self.data_size / 2)))
        random.shuffle(inds)
        trainSet, validSet, testSet = [], [], []
        for ele in inds[:nums[0]]:
            trainSet.append(self.pad_data[2 * ele - 1] + self.pad_data[2 * ele] + [self.PAD])
        for ele in inds[nums[0]:nums[1]]:
            validSet.append(self.pad_data[2 * ele - 1] + self.pad_data[2 * ele] + [self.PAD])
        for ele in inds[nums[1]:]:
            testSet.append(self.pad_data[2 * ele - 1] + self.pad_data[2 * ele] + [self.PAD])
        self.trainSet = np.array(trainSet).astype(np.int32)
        self.validSet = np.array(validSet).astype(np.int32)
        self.testSet = np.array(testSet).astype(np.int32)
    def shuffle_trainSet(self):
        l = len(self.trainSet)
        inds = list(range(l))
        random.shuffle(inds)
        self.trainSet = np.array([self.trainSet[i] for i in inds])
    def get_batch(self,n,batch_size):
        return self.trainSet[n*batch_size:(n+1)*batch_size,:],self.trainSet[n*batch_size:(n+1)*batch_size,self.pad_size:2*self.pad_size]


class tianya(data_set):
    def __init__(self,name='tianya'):
        data_set.__init__(self, name=name)

    def load(self, logitsDataPath, vocabPath, posDataPath=None,posDictPath=None):

        self.load_data(logitsDataPath,vocabPath)
        if posDictPath and posDataPath:

            self.load_POS(posDataPath,posDictPath)
        else:
            print('No POS data (>_<)')
    def build(self,jsonDataPath,txtDataPath=None,logitsDataPath=None,vocabPath=None,posDataPath=None,
              posDictPath=None,userdict=None,txtfilter=basic_filter):
        #extract text from json
        f = open(jsonDataPath, 'rb')
        d = f.readlines()
        f.close()
        if not txtDataPath:
            txtDataPath = 'data.txt'
        f = open(txtDataPath, 'w')
        for ele in d:
            j = json.loads(ele.decode('utf-8'))
            post = j['post'].replace('\n', '')
            # print(type(post))
            c = []
            for ele in j['cmnt']:
                c.append(ele['content'])
            for cms in c:
                cms = cms.replace('\n', '')
                pairs = post + '\n' + cms + '\n'
                if txtfilter(post, cms):
                    try:
                        f.write(pairs)
                    except:
                        continue
        f.close()
        #from text to logits
        if userdict:
            jieba.load_userdict(userdict)
        f = open(txtDataPath)
        d = f.readlines()
        f.close()
        vocab, poss = {'UNK':self.UNK,'START':self.START,'PAD':self.PAD,'END':self.END}, {}
        d_logits, dp_logits = [], []
        for ele in d:
            words = list(pseg.cut(ele.replace('\n', '')))
            s_logits, sp_logits = [], []
            for wp in words:
                w, p = wp
                try:
                    vocab[w] += 0
                except:
                    vocab[w] = len(vocab)
                try:
                    poss[p] += 0
                except:
                    poss[p] = len(poss)
                s_logits.append(str(vocab[w]))
                sp_logits.append(str(poss[p]))
                s_logits = [str(self.START)] + s_logits + [str(self.END)]
            d_logits.append(' '.join(s_logits))
            dp_logits.append(' '.join(sp_logits))
        if not logitsDataPath:
            logitsDataPath = 'data.logits'
        f = open(logitsDataPath, 'w')
        f.write('\n'.join(d_logits))
        f.close()
        if not posDataPath:
            posDataPath = 'pos.logits'
        f = open(posDataPath, 'w')
        f.write('\n'.join(dp_logits))
        f.close()
        if not vocabPath:
            vocabPath = 'vocab.json'
        f = open(vocabPath, 'w')
        f.write(json.dumps(vocab))
        f.close()
        if not posDictPath:
            posDictPath = 'pos.json'
        f = open(posDictPath, 'w')
        f.write(json.dumps(poss))
        f.close()
        print('Successfully,saved in:\n%s\n%s\n%s\n%s\n%s'%(txtDataPath,logitsDataPath,vocabPath,posDataPath,posDictPath))



if __name__=='__main__':
    data0 = tianya()
    #data0.load('data/data.logits','data/vocab.json','data/pos.logits','data/pos.json')
    #data0.build('data/minitest.txt','data/data.txt','data/data.logits','data/vocab.json','data/pos.logits','data/pos.json','data/newwords.txt')
    data0.load('data/data.logits','data/vocab.json','data/pos.logits','data/pos.json')
    data0.build_trainSet()
    #data = data_set()
    #data.load_data('data/data.logits','data/vocab.json')
    #data.load_POS('data/pos.logits','data/pos.json')


        