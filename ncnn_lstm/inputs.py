import os
import numpy as np
import tensorflow as tf
import json
import re
import random
import h5py

import ncnn_lstm as model


def load_word(path):
        
    input_file = open(path)
    word_lst = [line.rstrip('\n') for line in input_file.readlines()]
    words = dict((word, i) for i, word in enumerate(word_lst))
    rwords = dict(map(lambda t:(t[1],t[0]), words.items()))
    input_file.close()
    
    return words, rwords
 

class Generate(object):
    def __init__(self):
        
        self.words, self.rwords = load_word('/home/xuwenshen/data/big_data/2017_3_13/words')
        self.samples = h5py.File('/home/xuwenshen/data/big_data/2017_3_13/test.h5')
        
        self.words_size = len(self.words)
        self.source_len = 1000
        self.simplified_len = 150
        self.oseq_len = 200
        self.batch_size = 50
        self.pattern_symbol = re.compile(r'[。！；;（）]')
        self.pattern_defend = re.compile(r'被告|罪犯|犯|罪|非法|涉嫌')
        
        self.model  =  model.build_model(words_size=self.words_size, 
                                         embedding_size=200,
                                         source_len=1000,
                                         simplified_len=150,
                                         oseq_len=200, 
                                         decoder_hidden=750,
                                         source_nfilters=480,
                                         defendant_nfilters=75,
                                         source_width=3,
                                         defendant_width=3,
                                         lstm_layer=1, 
                                         batch_size=50, 
                                         is_train=False)
    
        
        self.cost=self.model['cost']
        self.words_prediction=self.model['words_prediction']
        self.source=self.model['source']
        self.defendant=self.model['defendant']
        self.defendant_length=self.model['defendant_length']
        self.decoder_inputs=self.model['decoder_inputs']
        self.loss_weights=self.model['loss_weights']
        self.keep_prob=self.model['keep_prob']
        self.sample_rate=self.model['sample_rate']
    
        tvar = tf.trainable_variables()
        for v in tvar:
            print (v.name)
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction = 0.1)
        session_conf = tf.ConfigProto(allow_soft_placement = True, gpu_options = gpu_option, log_device_placement = False)

        self.sess = tf.Session(config=session_conf)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, '/home/xuwenshen/2017_3_13/2cnn-lstm/model_v1/sample_rate-0.5-train_accu-0.000-train_cost-0.762-train_bleu-0.000-valid_accu-0.791-valid_cost-0.764-valid_bleu-0.690-model.ckpt-70')
        
    
    def deal_text(self, text):
        
        source_tmp = re.sub(self.pattern_symbol, '，', text)
        source_tmp = source_tmp.split('，')
    
        defendant_tx = ''
        
        def_candicate = []        
    
        for i in range(len(source_tmp)):
            if re.search(self.pattern_defend, source_tmp[i]):
                if len(source_tmp[i]) > 30:
                    def_candicate.append(source_tmp[i])
                else:
                    defendant_tx += source_tmp[i] + '，'
            
       
       
        for i in range(len(def_candicate)):
            defendant_tx += def_candicate[i] + '，'
       
        
        simplified_source_tx = ''
        simplified_defendant_tx = ''
        for i in range(len(text)):
            if text[i] in self.words:
                simplified_source_tx += text[i]

        for i in range(len(defendant_tx)):
            if defendant_tx[i] in self.words:
                simplified_defendant_tx += defendant_tx[i]

   
        simplified_source_tx = simplified_source_tx[: min(self.source_len, len(simplified_source_tx))]
        simplified_defendant_tx = simplified_defendant_tx[: min(self.simplified_len, len(simplified_defendant_tx))]
        
        return {'source' : simplified_source_tx, 'defendant' : simplified_defendant_tx}
    
    
    def t2i(self, text_dict):
        
        pad = self.words['pad']
        go = self.words['go']
        
        source = [pad for j in range(self.source_len)]
        defendant = [pad for j in range(self.simplified_len)]
        decoder_input = [go for j in range(self.oseq_len)]
        
        start = self.source_len - len(text_dict['source'])
        for j in range(start, self.source_len):
            source[j] = float(self.words[text_dict['source'][j-start]])
        
    
        start = self.simplified_len - len(text_dict['defendant'])
        for j in range(start, self.simplified_len):
            defendant[j] = float(self.words[text_dict['defendant'][j-start]])

            
        source_index = []
        defendant_index = []
        decoder_inputs_index = []
        defendant_length = []
        
        for i in range(self.batch_size):
            source_index.append(source)
            defendant_index.append(defendant)
            decoder_inputs_index.append(decoder_input)
            defendant_length.append(len(text_dict['defendant']))  


        return {'source_tx': text_dict['source'],
                'defendant_tx': text_dict['defendant'],
                'source':source_index, 
                'defendant':defendant_index, 
                'decoder_input':decoder_inputs_index,
                'defendant_length':defendant_length}
            
    
    def i2t(self, ilist):
        
        reason_tx = ''
        seq_finished = False

        for i in range(self.oseq_len):
            if self.rwords[ilist[-1][i]] == 'eos':
                reason_tx += '。'
                seq_finished = True
                break
            reason_tx += self.rwords[ilist[-1][i]]
            
        if seq_finished == False: reason_tx += '...'
                
        return reason_tx
    
 
    def query(self, x):
        
        text = self.deal_text(x)
        inputs = self.t2i(text)
        


        tword = self.sess.run(self.words_prediction, feed_dict={self.source:inputs['source'],
                                                                self.defendant:inputs['defendant'],
                                                                self.decoder_inputs:inputs['decoder_input'],
                                                                self.defendant_length:inputs['defendant_length'],
                                                                self.sample_rate:1.,
                                                                self.keep_prob:1.})
        
        prediction_tx = self.i2t(tword)
        
        return {'source_tx': x, 'reason_tx': '', 'prediction_tx': prediction_tx}
    
    def sample(self):
        
        ids = random.randint(50, 100000)
        
        source_ = self.samples['source'][ids]
        defendant_ = self.samples['defendant'][ids]
        decoder_inputs_ = self.samples['ground_truth'][ids]
        defendant_length_ = self.samples['defendant_length'][ids]
        
        source_index = []
        defendant_index = []
        decoder_inputs_index = []
        defendant_length = []
        
        for i in range(self.batch_size):
            source_index.append(source_)
            defendant_index.append(defendant_)
            decoder_inputs_index.append(decoder_inputs_)
            defendant_length.append(defendant_length_)  

        tword = self.sess.run(self.words_prediction, feed_dict={self.source:source_index,
                                                                self.defendant:defendant_index,
                                                                self.decoder_inputs:decoder_inputs_index,
                                                                self.defendant_length:defendant_length,
                                                                self.sample_rate:1.,
                                                                self.keep_prob:1.})
        prediction_tx = self.i2t(tword)


        return {'source_tx': self.samples['source_tx'][ids].decode('gb2312'),
                'reason_tx': self.samples['reason_tx'][ids].decode('gb2312'), 
                'prediction_tx':prediction_tx}


if __name__ == '__main__':
    
    generate = Generate()
    
    while True:
       
        print ('Input Source: ') 
        raw_text = input()
        
        if raw_text == 'random':
            tx = generate.sample()
            
            print ('\n' +  '+' * 20 + '\n')
            
            print (tx['source_tx'])
            print ('-' * 20 + '\n')

            print (tx['reason_tx'])
            print ('-' * 20 + '\n')

            print (tx['prediction_tx'])
            print ('*' * 20 + '\n')
            
            print ('\n')
            
        else:
            tx = generate.query(raw_text)
            
            print (tx['source_tx'])
            print ('-' * 20 + '\n')

            print (tx['reason_tx'])
            print ('-' * 20 + '\n')

            print (tx['prediction_tx'])
            print ('*' * 20 + '\n')




       
       

