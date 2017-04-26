import tensorflow as tf
import numpy as np

class seq2seq(object):
    def __init__(self,emb_dim=16,vocab_size=101,encoder_size=5,decoder_size=5,lr=0.002,
                 forward_only=False,cell=tf.contrib.rnn.LSTMCell,num_units=128,name='seq2seq'):
        self.name = name
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.decoder_size = decoder_size
        self.encoder_size = encoder_size
        cell = cell(num_units)
        self.inputs = tf.placeholder(tf.int32, shape=[None, encoder_size + decoder_size + 1], name='inputs')
        self.targets = tf.placeholder(tf.int32,shape=[None,decoder_size],name='targets')
        with tf.variable_scope(self.name):
            embeddings = tf.get_variable(name='embeddings', shape=[self.vocab_size,emb_dim],
                                         initializer=tf.random_uniform_initializer())
            w_proj = tf.get_variable(name='w_proj',shape=[num_units,vocab_size],initializer=tf.random_uniform_initializer())
            b_proj = tf.get_variable(name='b_proj',shape=[vocab_size],initializer=tf.random_uniform_initializer())
            print('inputs shape:',self.inputs.get_shape())
            emb_inputs = tf.nn.embedding_lookup(embeddings,self.inputs)
            print('emb_inputs shape:',emb_inputs.get_shape())
            emb_inputs = tf.transpose(emb_inputs,[1,0,2])#[batch,step,emb_size]-->[step,batch,emb_size]
            emb_inputs = tf.unstack(emb_inputs)
            _outputs,_ = tf.contrib.rnn.static_rnn(cell,emb_inputs,dtype=tf.float32)

        self.outputs = [tf.matmul(ele,w_proj)+b_proj for ele in _outputs[encoder_size+1:]]#step*[batch,emb_dims]
        self.outputs = tf.concat(self.outputs,axis=0)#[]
        #targets_one_hot = tf.one_hot(self.targets,vocab_size,1,0)
        #print('toh shape:',targets_one_hot.get_shape())
        #targets_ = tf.reshape(targets_one_hot,[-1,vocab_size])
        targets_ = tf.reshape(self.targets,[-1])
        print('targets_ shape:',targets_.get_shape())
        print('outputs shape:',self.outputs.get_shape())
        print('targets shape:',self.targets.get_shape())
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(targets_,self.outputs))
        self.opt = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)
    def train(self,data,batch_size,max_epoch,save_step=10,display_step=10,save2='model/model_seq2seq.ckpt'):
        num_steps = data.data_size//batch_size
        sess = tf.Session()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)
        for i in range(max_epoch):
            step = 0
            while step<num_steps:
                source,target = data.get_batch(step,batch_size)
                #_inputs = np.column_stack((source,target))
                feed_dict = {self.inputs.name:source,self.targets.name:target}
                cost,_ = sess.run([self.loss,self.opt],feed_dict=feed_dict)
                if step%display_step == 0:
                    print('epoch:%d,step:%d,cost:%f'%(i,step,cost))
                if step%save_step == 0:
                    saver.save(sess,save2)
                    print('model saved in %s'%save2)
                step += 1
	    print('Do test')
            source, target = data.get_testSet()
            feed_dict = {self.inputs.name:source,self.targets.name:target}
            outp, cost = sess.run(self.outputs,feed_dict=feed_dict)
            outp = np.argmax(outp.reshape([-1,data.pad_size]),axis=1)
            outp = outp.reshape([-1,data.pad_size])
            query = data.logits2sentence(source)
            real_resp = data.logits2sentence(target)
            pred_resp = data.logits2sentence(outp)
	    for i in range(outp.shape[0]):
                print('Query:%s\nRResp:%s\nPResp:%s\n'%(query[i],real_resp[i],pred_resp[i]))
            data.shuffle_trainSet()

class test_data:
    def __init__(self,sources,targets):
        self.sources = sources
        self.targets = targets
        self.size = len(self.sources)
    def get_batch(self,n,batch_size):
        return self.sources[n*batch_size:(n+1)*batch_size,:],self.targets[n*batch_size:(n+1)*batch_size,:]
if __name__=='__main__':
    np.random.seed(1)
    n_samples = 100000
    data_x = np.random.randint(1,101,[n_samples,10],np.int32)
    print(data_x.shape)
    data_y = data_x[:,5:]
    print(data_y.shape)
    data_x = np.column_stack((data_x,np.zeros([n_samples,1],np.int32)))
    data = test_data(data_x,data_y)
    model = seq2seq()
    model.train(data,10,10)

