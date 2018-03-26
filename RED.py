import tensorflow as tf
import numpy as np

"""
NOTE!

This code will not run, key variables are undeclared and this is half-baked af
Things to consider:
-This has a large portion of relevant functions and classes for our implementation
-How do we want to structure it? (i.e how is training done? How is inference done? How do we get evals mid train?)
-How much do we want to use the API vs how low level do we want to go.
-The API has a neat structure and even this right now messes with it

Worst case we can copy the seq2seq model they have completely and then modify the relevant parts
and hack it into a VHRED

"""

class RED():
    def __init__(self, enc_embeds, dec_embeds):
        
        #what other settings are important for initialization?
        
        self.enc_cell = tf.nn.rnn_cell.BasicRNNCell(1)
        self.dec_cell = tf.nn.rnn_cell.BasicRNNCell(1)
        self.enc_embeds = enc_embeds #should these be classes implemented in processing?
        self.dec_embeds = dec_embeds
        self.V #SET vocab size!!
    
    def train(self, encoder_inp, decoder_inp):
        
        seq_length = encoder_inp.shape[1] #Is this the right dimension for the sequence length?
        
        #build encoder output node (in computational graph) and state
        encoder_outs, encoder_state = tf.nn.dynamic_rnn(
            self.enc_cell,
            encoder_inp,
            sequence_length=seq_length,
            time_major=True)
        
        #More or less copied for now. Do we want to use this helper in the future? I think not
        
        
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_inp, decoder_lengths, time_major=True) #decoder_inp, decoder_lengths UNDECLARED!
        output_dense = tf.layers.core.Dense(
            self.V, use_bias=False)         #Vocab size uninitialized!
        decoder = tf.contrib.seq2seq.BasicDecoder(
            self.dec_cell, helper, encoder_state, output_layer)
        
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True)
        logits = outputs.rnn_output
        
        #the remainder is copied due to time constraints
        
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=decoder_outputs, logits=logits) #decoder outputs??
        
        train_loss = (tf.reduce_sum(crossent * target_weights) /batch_size) #batch size??
        
        params = tf.trainable_variables()
        gradients = tf.gradients(train_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, max_gradient_norm)
        
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_step = optimizer.apply_gradients(
            zip(clipped_gradients, params))
        
        
    

    