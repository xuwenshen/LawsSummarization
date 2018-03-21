import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell, seq2seq
from tqdm import *
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import sys


def construct_data(words_size, embedding_size, source_len, oseq_len, decoder_hidden, encoder_hidden, 
                   simplified_len, source_nfilters, source_width,):
    
    
    embedding = tf.Variable(tf.random_uniform([words_size, embedding_size], -1., 1.), name = 'embedding')

    source = tf.placeholder(tf.int32, [None, source_len], name = 'source')
    
    defendant = tf.placeholder(tf.int32, [None, simplified_len], name = 'defendant')
    
    label = tf.placeholder(tf.int32, [None, oseq_len], name = 'label')

    decoder_inputs = tf.placeholder(tf.int32, [None, oseq_len], name = 'decoder_inputs')
    
    defendant_length = tf.placeholder(tf.int32, [None], name = 'defendant_length')

    loss_weights = tf.placeholder(tf.float32, [None, oseq_len], name = 'loss_weights')
    
    sample_rate = tf.placeholder(tf.float32, shape = (), name = 'sample_rate')
    
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')


    conv_args = {
        'conv_w' : {
            'conv1': tf.Variable(tf.random_uniform([source_width, embedding_size, 1, source_nfilters], -.01, .01),
                                 name='conv_source_w_1'),
            'conv2': tf.Variable(tf.random_uniform([source_width, 1, source_nfilters, source_nfilters], -.01, .01),
                                 name='conv_source_w_2'),
            'conv3': tf.Variable(tf.random_uniform([source_width, 1, source_nfilters, source_nfilters], -.01, .01),
                                 name='conv_source_w_3')      
        },

        'conv_b' : {
            'conv1': tf.Variable(tf.zeros([source_nfilters]), name='conv_source_b_1'),
            'conv2': tf.Variable(tf.zeros([source_nfilters]), name='conv_source_b_2'),
            'conv3': tf.Variable(tf.zeros([source_nfilters]), name='conv_source_b_3')                
        },

        'scale'  : {
            'conv1': tf.Variable(tf.ones([source_nfilters]), name='conv_source_scale_1'),
            'conv2': tf.Variable(tf.ones([source_nfilters]), name='conv_source_scale_2'),
            'conv3': tf.Variable(tf.ones([source_nfilters]), name='conv_source_scale_3')  
        },

        'offset' : {
            'conv1': tf.Variable(tf.zeros([source_nfilters]), name='conv_source_offset_1'),
            'conv2': tf.Variable(tf.zeros([source_nfilters]), name='conv_source_offset_2'),
            'conv3': tf.Variable(tf.zeros([source_nfilters]), name='conv_source_offset_3') 
        },

        'moving_avg': {
            'conv1': tf.Variable(tf.zeros([source_nfilters]), name = "conv_source_mavg_1", trainable=False),
            'conv2': tf.Variable(tf.zeros([source_nfilters]), name = "conv_source_mavg_2", trainable=False),
            'conv3': tf.Variable(tf.zeros([source_nfilters]), name = "conv_source_mavg_3", trainable=False)
        },

        'moving_var' : {
            'conv1': tf.Variable(tf.zeros([source_nfilters]), name = "conv_source_mvar_1", trainable=False),
            'conv2': tf.Variable(tf.zeros([source_nfilters]), name = "conv_source_mvar_2", trainable=False),
            'conv3': tf.Variable(tf.zeros([source_nfilters]), name = "conv_source_mvar_3", trainable=False)
        }
    }
            
    weigth_generation = tf.Variable(tf.random_uniform([decoder_hidden, words_size], -.01, .01), name='generation_w')
 
    bias_generation = tf.Variable(tf.zeros([words_size]), name = 'generation_b')
    
    

    
    return {'embedding':embedding,
            'conv_args':conv_args,
            'weigth_generation':weigth_generation, 
            'bias_generation':bias_generation,
            'source':source,
            'defendant':defendant,
            'defendant_length':defendant_length,
            'label':label, 
            'sample_rate':sample_rate,
            'decoder_inputs':decoder_inputs,
            'loss_weights':loss_weights,
            'keep_prob':keep_prob}

    
def batch_normalizarion(x, offset, scale, mavg, mvar, is_train):
    
    control_inputs = []
    if is_train:
        avg, var = tf.nn.moments(x=x, axes=list(range(len(x.get_shape()) - 1)))
        update_moving_avg = moving_averages.assign_moving_average(mavg, avg, decay=0.5)
        update_moving_var = moving_averages.assign_moving_average(mvar, var, decay=0.5)
        control_inputs = [update_moving_avg, update_moving_var]
    else:
        avg = mavg
        var = mvar
    with tf.control_dependencies(control_inputs):
        return tf.nn.batch_normalization(x=x,
                                         mean=avg, 
                                         variance=var, 
                                         offset=offset,
                                         scale=scale, 
                                         variance_epsilon=0.001)


def conv2d(x, w, b, offset, scale, mavg, mvar, strides=1, name='conv', padding='VALID', is_train=True):
    
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding, name=name)
    x = tf.nn.bias_add(x, b)
    x = batch_normalizarion(x, offset, scale, mavg, mvar, is_train)
    return tf.nn.relu(x)

    

def maxpool(x, k1=1, k2=1, name='pooling', padding='VALID'):
    
    return tf.nn.max_pool(x, ksize=[1, k1, k2, 1], strides=[1, 1, 1, 1], padding=padding, name=name)



def encoder_conv(source, defendant, conv_args, keep_prob, embedding, is_train):
    
    with tf.name_scope('encoder_conv') as scope:
        
        embedding_size = embedding.get_shape().as_list()[-1]

        batch_x = tf.nn.embedding_lookup(embedding, source)
        batch_x = tf.reshape(batch_x, shape=[-1, source.get_shape().as_list()[-1], embedding_size, 1])

        conv1 = conv2d(x=batch_x, 
                       offset=conv_args['offset']['conv1'], 
                       scale=conv_args['scale']['conv1'], 
                       w=conv_args['conv_w']['conv1'], 
                       b=conv_args['conv_b']['conv1'],
                       mavg=conv_args['moving_avg']['conv1'],
                       mvar=conv_args['moving_var']['conv1'],
                       is_train=is_train)
        print (conv1.get_shape().as_list())


        conv2 = conv2d(x=conv1, 
                       offset=conv_args['offset']['conv2'], 
                       scale=conv_args['scale']['conv2'], 
                       w=conv_args['conv_w']['conv2'], 
                       b=conv_args['conv_b']['conv2'],
                       mavg=conv_args['moving_avg']['conv2'],
                       mvar=conv_args['moving_var']['conv2'],
                       is_train=is_train)
        print (conv2.get_shape().as_list())


        conv3 = conv2d(x=conv2, 
                       offset=conv_args['offset']['conv3'], 
                       scale=conv_args['scale']['conv3'], 
                       w=conv_args['conv_w']['conv3'], 
                       b=conv_args['conv_b']['conv3'],
                       mavg=conv_args['moving_avg']['conv3'],
                       mvar=conv_args['moving_var']['conv3'],
                       is_train=is_train)
        print (conv3.get_shape().as_list())


        pool = maxpool(x=conv3, k1=conv3.get_shape().as_list()[1], k2=conv3.get_shape().as_list()[2])
        pool = tf.reshape(pool, [-1, pool.get_shape().as_list()[-1]])

        output = tf.nn.dropout(pool, keep_prob)  

        return tf.concat(1, output)



def encoder_rnn(defendant, defendant_length, encoder_hidden, embedding, keep_prob, batch_size):
    
    with tf.name_scope('encoder_rnn') as scope:
        
        lstm_cell = rnn_cell.BasicLSTMCell(encoder_hidden, forget_bias=1.0, state_is_tuple=True)

        initial_state = lstm_cell.zero_state(batch_size, tf.float32)

        inputs = tf.reverse(defendant, dims=[False, True])

        inputs = tf.nn.embedding_lookup(embedding, inputs)

        outputs, states = tf.nn.dynamic_rnn(cell = lstm_cell, 
                                            inputs = inputs, 
                                            initial_state=initial_state, 
                                            sequence_length=defendant_length,
                                            time_major=False,
                                            scope='dynamic_rnn_encoder')

        return tf.nn.dropout(outputs, keep_prob), states




def decoder_rnn(conv_encoder, rnn_encoder, decoder_inputs, decoder_hidden, weigth_generation, 
                n_steps, bias_generation, batch_size, keep_prob,
                defendant, embedding, sample_rate, lstm_layer=1, is_train=True):
    
    with tf.name_scope('decoder_rnn') as scope:
        
        lstm_cell = rnn_cell.BasicLSTMCell(decoder_hidden, forget_bias=1.0, state_is_tuple=True)

        if lstm_layer > 1:
            lstm_cell = rnn_cell.MultiRNNCell([lstm_cell] * lstm_layer)
           
        initial_state = lstm_cell.zero_state(batch_size, tf.float32)
        
        batch_decoder_inputs = tf.nn.embedding_lookup(embedding, decoder_inputs)
        batch_decoder_inputs = tf.transpose(batch_decoder_inputs, [1, 0, 2])
        batch_decoder_inputs = tf.unpack(batch_decoder_inputs)
        batch_decoder_inputs = [tf.concat(1, [batch_decoder_inputs[i], conv_encoder]) 
                                for i in range(len(batch_decoder_inputs))]


        if is_train:
            def func(prev, i):

                #words prob
                words_prob = tf.nn.bias_add(tf.matmul(prev, weigth_generation), bias_generation)
                    
                sample = tf.argmax(words_prob, 1)
                prev_word = tf.nn.embedding_lookup(embedding, sample)
                prev_outputs = tf.concat(1, [prev_word, conv_encoder])


                # select from prev_outputs and ground truth
                prob =  tf.random_uniform(minval=0, maxval=1, shape=(batch_size,))
                mask = tf.cast(tf.greater(sample_rate, prob), tf.float32)
                mask = tf.expand_dims(mask, 1)
                mask = tf.tile(mask, [1, prev_outputs.get_shape().as_list()[-1]])

                next_input = mask * prev_outputs + (1 - mask) * batch_decoder_inputs[i]

                return next_input

            outputs, state = seq2seq.attention_decoder(decoder_inputs=batch_decoder_inputs,
                                                       initial_state=initial_state,
                                                       attention_states=rnn_encoder,
                                                       cell=lstm_cell,
                                                       num_heads=1,
                                                       loop_function=func,
                                                       scope='rnn_decoder',
                                                       initial_state_attention=False)



        else:

            def func(prev, i):

                #words prob
                words_prob = tf.nn.bias_add(tf.matmul(prev, weigth_generation), bias_generation)

                sample = tf.argmax(words_prob, 1)
                prev_word = tf.nn.embedding_lookup(embedding, sample)
                prev_outputs = tf.concat(1, [prev_word, conv_encoder])
                
                return prev_outputs

            outputs, state = seq2seq.attention_decoder(decoder_inputs=batch_decoder_inputs,
                                                       initial_state=initial_state,
                                                       attention_states=rnn_encoder,
                                                       cell=lstm_cell,
                                                       num_heads=1,
                                                       loop_function=func,
                                                       scope='rnn_decoder',
                                                       initial_state_attention=False)
                                        

        outputs = tf.nn.dropout(outputs, keep_prob)
        outputs = tf.unpack(outputs)

        res = [0 for i in range(n_steps)]
        for i in range(len(outputs)):
            #words prob
            res[i]= tf.nn.bias_add(tf.matmul(outputs[i], weigth_generation), bias_generation)

        return res, state

    
def build_model(words_size, embedding_size, oseq_len, source_len, simplified_len,
          encoder_hidden, decoder_hidden, lstm_layer, batch_size, source_nfilters, source_width, is_train):    
    
    args = construct_data(words_size=words_size, 
                          embedding_size=embedding_size,
                          source_len=source_len,
                          simplified_len=simplified_len,
                          oseq_len=oseq_len, 
                          encoder_hidden=encoder_hidden,
                          decoder_hidden=decoder_hidden,
                          source_nfilters=source_nfilters,
                          source_width=source_width)
    
    embedding = args['embedding']
    conv_args=args['conv_args']
    weigth_generation = args['weigth_generation']
    bias_generation = args['bias_generation']
    source = args['source']
    defendant = args['defendant']
    defendant_length = args['defendant_length']
    label = args['label']
    decoder_inputs = args['decoder_inputs']
    loss_weights = args['loss_weights']
    keep_prob = args['keep_prob']
    sample_rate = args['sample_rate']
    
    
    conv_encoder = encoder_conv(source=source,
                                defendant=defendant,
                                conv_args=conv_args,
                                keep_prob=keep_prob,
                                embedding=embedding,
                                is_train=is_train)
    
    rnn_encoder, encoder_states = encoder_rnn(defendant=defendant,
                                              defendant_length=defendant_length,
                                              encoder_hidden=encoder_hidden, 
                                              keep_prob=keep_prob,
                                              batch_size=batch_size,
                                              embedding=embedding)

    rnn_decoder, state_decoder = decoder_rnn(conv_encoder=conv_encoder,
                                             rnn_encoder=rnn_encoder,
                                             defendant=defendant,
                                             decoder_inputs=decoder_inputs,
                                             decoder_hidden=decoder_hidden, 
                                             weigth_generation=weigth_generation,
                                             bias_generation=bias_generation,
                                             n_steps=oseq_len,
                                             batch_size=batch_size, 
                                             lstm_layer=lstm_layer, 
                                             keep_prob=keep_prob,
                                             embedding=embedding,
                                             sample_rate=sample_rate,
                                             is_train=is_train)

    
    cost = tf.reduce_mean(seq2seq.sequence_loss_by_example(logits=rnn_decoder,
                                                           targets=tf.unpack(tf.transpose(label, [1,0])),
                                                           weights=tf.unpack(tf.transpose(tf.convert_to_tensor(
                                                               loss_weights, dtype=tf.float32), [1,0]))))
    
    
    words_prediction = tf.argmax(tf.transpose(tf.pack(rnn_decoder), [1, 0, 2]), 2)
    
    
    print ('build model ')
    
    
    return {'outputs':rnn_decoder, 
            'embedding':embedding,
            'cost':cost,
            'sample_rate':sample_rate,
            'words_prediction':words_prediction,
            'source':source,
            'defendant':defendant,
            'defendant_length':defendant_length,
            'label':label, 
            'decoder_inputs':decoder_inputs, 
            'loss_weights':loss_weights, 
            'keep_prob':keep_prob}




