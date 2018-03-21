import re, os, sys, json, random
from tqdm import *
import numpy as np
import tensorflow as tf  
import h5py
import nltk


def load_word(path):
        
    input_file = open(path)
    word_lst = [line.rstrip('\n') for line in input_file.readlines()]
    words = dict((word, i) for i, word in enumerate(word_lst))
    rwords = dict(map(lambda t:(t[1],t[0]), words.items()))
    input_file.close()
    
    return words, rwords


class Utils:
    
    def __init__(self, word_path, text_path, batch_size, nb_samples):
        
        self.words, self.rwords = load_word(word_path)
        self.file = h5py.File(text_path)
        self.batch_size = batch_size
        self.current_batch = 0
        self.nb_samples = nb_samples
        self.current_text = dict()
        self.shuffled_id = np.arange(nb_samples)
        random.shuffle(self.shuffled_id)
        
    
    def get_words_size(self):
        return len(self.words)
    
        
    def next_batch(self):
        
        to_again = False
        if (self.current_batch + 1) * self.batch_size >= self.nb_samples:
            to_again = True
            random.shuffle(self.shuffled_id)
            self.current_batch = 0
        
        if to_again:
            return dict(), to_again
        
        start = self.current_batch * self.batch_size
        end = (self.current_batch + 1) * self.batch_size
        ids = self.shuffled_id[start:end]
        ids = sorted(ids)
        
        source = []
        ground_truth = []
        label = []
        loss_weights = []
        defendant = []
        defendant_length = []
        
        source_tx = []
        defendant_tx = []
        reason_tx = []
        
        
        source = self.file['source'][ids]
        ground_truth = self.file['ground_truth'][ids]
        label = self.file['label'][ids]
        loss_weights = self.file['loss_weights'][ids]
        defendant = self.file['defendant'][ids]
        defendant_length = self.file['defendant_length'][ids]
        
        
        source_tx = self.file['source_tx'][ids]
        defendant_tx = self.file['defendant_tx'][ids]
        reason_tx = self.file['reason_tx'][ids]
            
                                              
        
        to_return = {'source' : source,
                     'defendant' : defendant,
                     'defendant_length' : defendant_length,
                     'ground_truth' : ground_truth, 
                     'label' : label,
                     'loss_weights' : loss_weights}
        
        self.current_text = to_return
        self.current_text.update({'source_tx':source_tx, 'defendant_tx':defendant_tx, 'reason_tx':reason_tx})
        self.current_batch += 1
        
        return to_return, to_again   
    
    
    def print_text(self, prediction_tx, index):


        print (self.current_text['source_tx'][index].decode('gb2312'))
        print ('-' * 20 + '\n')
        
        print (self.current_text['defendant_tx'][index].decode('gb2312'))
        print ('-' * 20 + '\n')
        
        print (self.current_text['reason_tx'][index].decode('gb2312'))
        print ('-' * 20 + '\n')
        
        print (prediction_tx)
        print ('\n' + '*' * 20 + '\n')
    
    def bleu(self, prediction_tx, index):
        
        return nltk.translate.bleu_score.sentence_bleu([self.current_text['reason_tx'][index].decode('gb2312')], prediction_tx)
    
    def i2t(self, ilist, to_print):
        
        same_words_counter = 0
        words_counter = 0
        
        bleu_score = 0
        
        for i in range(len(ilist)):
                
            prediction_tx = ''
            
            for j in range(len(ilist[i])):
                
                if self.rwords[self.current_text['label'][i][j]] == 'pad': 
                    break
                
                words_counter += 1
                if self.current_text['label'][i][j] == ilist[i][j]:
                    same_words_counter += 1
            
            
                
            for j in range(len(ilist[i])):
                if self.rwords[ilist[i][j]] == 'eos':
                    break
                prediction_tx += self.rwords[ilist[i][j]]
         
            bleu_score += self.bleu(prediction_tx, i)
            
            if i != len(ilist)-1: continue
                
            if to_print:
                self.print_text(prediction_tx=prediction_tx,  index=i)

        return same_words_counter / words_counter, bleu_score / len(ilist)
            

    
    