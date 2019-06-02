import tensorflow as tf
import numpy as np
from config import *
class Lstm_model():
    def __init__(self):
        self.truncate_l=truncate_l
        self.vocab_size=vocab_size
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.zuiming_number=zuiming_number
        self.keep_rate = keep_rate

        with tf.name_scope('Input'):
            self.test_zuiming_label=tf.placeholder(shape=(None,),dtype=tf.int64)
            self.fzss=tf.placeholder(shape=(None,self.truncate_l),dtype=tf.int64)
            self.add_inf=tf.placeholder(shape=(None, 3),dtype=tf.float32)
            self.all_label_c=tf.placeholder(shape=(None, 21),dtype=tf.int64)
            self.all_label_r = tf.placeholder(shape=(None, 1), dtype=tf.int64)
            self.input_length=tf.placeholder(tf.int64,(None))

        with tf.name_scope('Lstm'):
            self.embedding_talble=tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_size],dtype=tf.float32),trainable=True,name='word_embedding')
            emb_input=tf.nn.embedding_lookup(self.embedding_talble, self.fzss)
            cell_f=tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            cell_b = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            output,_=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_f,cell_bw=cell_b,inputs=emb_input,sequence_length=self.input_length,dtype=tf.float32)
            output_concate_bid=tf.concat(output, axis=-1)

        with tf.name_scope('Attention'):
            output_concate=tf.keras.layers.Dense(self.hidden_size*2)(output_concate_bid)
            attention_list=[]
            for i in range(20):
                att_vec=tf.get_variable(name='att_vec'+str(i),shape=(1,self.hidden_size*2),dtype=tf.float32,trainable=True)
                att_rate_vec=tf.reduce_sum(tf.multiply(output_concate,att_vec),keep_dims=True,axis=2)
                att_rate_vec=tf.nn.softmax(att_rate_vec)
                att_sum=tf.reduce_sum(tf.multiply(output_concate,att_rate_vec),axis=1)
                attention_list.append(att_sum)
            JJJ=tf.expand_dims(attention_list[0],axis=1)
            for j in range(1,20):
                tocancate=tf.expand_dims(attention_list[j],axis=1)
                JJJ=tf.concat([JJJ,tocancate],axis=1)
            JJJ = tf.expand_dims(JJJ, axis=3)
            avage_pool_vec=tf.nn.avg_pool(JJJ,ksize=[1,20,1,1],strides=[1,1,1,1],padding='VALID')
            avage_pool_vec=tf.reduce_sum(avage_pool_vec,axis=1)
            avage_pool_vec = tf.reduce_sum(avage_pool_vec, axis=-1)

        with tf.name_scope('max_pool'):
            output_concate_bid = tf.expand_dims(output_concate_bid, axis=-1)
            output_concate_bid=tf.nn.max_pool(output_concate_bid,ksize=(1,300,1,1),strides=(1,1,1,1),padding='VALID')
            output_concate_bid=tf.reduce_sum(output_concate_bid,axis=-1)
            output_concate_bid = tf.reduce_sum(output_concate_bid, axis=1)
            output_feature4zuiming=tf.concat([avage_pool_vec,output_concate_bid],axis=-1)

        with tf.name_scope('classifier'):
            layer1 = tf.get_variable(name='c1layer1', shape=(self.hidden_size*4, 80), dtype=tf.float32)
            b1 = tf.get_variable(name='c1b1', shape=(80), dtype=tf.float32)
            output_layer1 = tf.nn.leaky_relu(tf.nn.xw_plus_b(output_feature4zuiming, layer1, b1))
            layer2 = tf.get_variable(name='c1layer2' , shape=(80, self.zuiming_number), dtype=tf.float32)
            b2 = tf.get_variable(name='c1b2', shape=(self.zuiming_number), dtype=tf.float32)
            output_layer2 = tf.nn.leaky_relu(tf.nn.xw_plus_b(output_layer1, layer2, b2))

            self.output_list=[output_layer2]
            for i in range(20):
                layer1 = tf.get_variable(name='c'+str(i+2)+'layer1', shape=(self.hidden_size*2, 40), dtype=tf.float32)
                b1 = tf.get_variable(name='c'+str(i+2)+'b1', shape=(40), dtype=tf.float32)
                output_layer1 = tf.nn.leaky_relu(tf.nn.xw_plus_b(attention_list[i], layer1, b1))
                layer2 = tf.get_variable(name='c'+str(i+2)+'layer2', shape=(40, 2), dtype=tf.float32)
                b2 = tf.get_variable(name='c'+str(i+2)+'b2' + str(i), shape=(2), dtype=tf.float32)
                output_layer2 = tf.nn.leaky_relu(tf.nn.xw_plus_b(output_layer1, layer2, b2))
                self.output_list.append(output_layer2)

        with tf.name_scope('xingqi'):
            output_feture4xingqi=tf.concat([output_feature4zuiming,self.add_inf],axis=-1)
            layer1 = tf.get_variable(name='xingqiL1' , shape=(output_feture4xingqi.shape[1],80 ), dtype=tf.float32)
            b1 = tf.get_variable(name='xingqiB1' , shape=(80), dtype=tf.float32)
            output_layer1 = tf.nn.leaky_relu(tf.nn.xw_plus_b(output_feture4xingqi, layer1, b1))
            layer2 = tf.get_variable(name='xingqiL2' , shape=(80, 11), dtype=tf.float32)
            b2 = tf.get_variable(name='xingqiB2' , shape=(11), dtype=tf.float32)
            output_layer2 = tf.nn.leaky_relu(tf.nn.xw_plus_b(output_layer1, layer2, b2))
            self.xingqi_pre=tf.argmax(output_layer2,axis=1)
            self.output_list.append(output_layer2)

        with tf.name_scope('loss'):
            for i in range(21):
                label=tf.slice(self.all_label_c,[0,i],[self.batch_size,1])
                label=tf.reshape(label,[label.shape[0]])
                loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output_list[i],labels=label))
                tf.add_to_collection('losses',loss)
            label =tf.reduce_mean(self.all_label_r,axis=-1)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output_list[-1], labels=label))
            tf.add_to_collection('losses', loss)
            losses=tf.add_n(tf.get_collection('losses'))
            self.loss=losses

        with tf.name_scope('acc'):
            max_index=tf.argmax(self.output_list[0],axis=1)
            equal_zhi = tf.cast(tf.equal(max_index, self.test_zuiming_label), dtype=tf.float32)
            self.acc =tf.reduce_mean(equal_zhi,axis=-1)

        with tf.name_scope('xingqi_acc'):
            xq_index = tf.argmax(self.output_list[-1],axis=1)
            label_r = tf.reduce_mean(self.all_label_r,axis=-1)
            xq_equal_zhi = tf.cast(tf.equal(xq_index,label_r),dtype=tf.float32)
            self.xq_acc = tf.reduce_mean(xq_equal_zhi,axis=-1)

# model = Lstm_model()
# init_op=tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     sess.run(model.loss)
