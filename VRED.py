from keras import backend as K
from keras.layers import Dense, Input, RNN, SimpleRNNCell, Lambda
from keras.models import Model
from keras.losses import categorical_crossentropy
import numpy as np
import processing
from argparse import ArgumentParser
import pickle as pkl

#SET HPARAMS MANUALLY BEFORE RUNNING
#hparams = [input_shape, h_size, dictionary_size, last_layer_activ, batch_size]

input_shape = (10,300)
h_size = 300
dictionary_size = 3002
last_layer_activ = 'softmax'
batch_size = 1
dict_shape = (10,3002)
output_shape = (10,300)

#Helper function for sampling from latent distribution
def sample(args):
    z_mean, z_var = args
    epsilon = K.random_normal(shape=(batch_size, h_size))
    z = z_mean + epsilon*K.exp(0.5*z_var)
    return z



RNN_cell = SimpleRNNCell(h_size)
#inputs
encoder_inputs = Input(shape= input_shape, name='encoder_inputs')
decoder_inputs = Input(shape= output_shape, name = 'decoder_inputs')
#decoder_targets = Input(shape= dict_shape, name = 'decoder_targets') #IS this necessary?

#encoder
encoder = RNN(RNN_cell, return_state=True)
#encoder_outputs only for debugging
encoder_outputs, h_final = encoder(encoder_inputs)

#latent distribution reparameterization
z_mean = Dense(h_size)(h_final)
z_var = Dense(h_size)(h_final)
z = Lambda(sample, output_shape=(h_size,))([z_mean, z_var])

#decoder
decoder = RNN(RNN_cell, return_sequences=True)
decoder_outputs = decoder(decoder_inputs, initial_state = z)
decoder_dense = Dense(dictionary_size, activation=last_layer_activ)
decoder_output_probs = decoder_dense(decoder_outputs)

#Build model
vred = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='VRED')

dummy_x_train = np.random.randn(10,10,300)
dummy_y_train = np.random.randn(10,10,300)
dummy_decoder_train = np.random.randn(10,5,3000)
one_hot_vectors,Output,Input = processing.main()

epochs = 4
if __name__ == '__main__':
    
    #parser = ArgumentParser()
    #parser.add_argument('-c', '--config')
    #args = parser.parse_args()
    #assert(args.config)
    #pkl.loads(args.config)
    
    #loss
    KL_loss = -0.5*K.sum(K.mean(1 + z_var - K.square(z_mean) - K.exp(z_var)))
    loss = categorical_crossentropy(decoder_targets, decoder_output_probs) + KL_loss
    vred.add_loss(loss)
    vred.compile(optimizer='adam')
    
    vred.summary()
    
    vred.l
    
    #last_layer_activ = 'sigmoid'
    
    vred.fit([np.array(Input),np.array(Output)],np.array(one_hot_vectors),epochs=epochs, batch_size=batch_size)
    
        
