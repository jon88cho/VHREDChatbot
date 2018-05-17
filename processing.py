import pandas as pd
from gensim.models import KeyedVectors
import numpy as np
from gensim.corpora import Dictionary
from keras.preprocessing import sequence
#declare global variables

Start_vector = np.zeros(300,)
Unk_vector = np.random.randn(300,)
Padding_vector = np.random.randn(300,)
keep_n = 3000 #Size of dictionary
word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)



#function to split text data, drop punctuation
def splitting(text):
    words = text.split()
    table = str.maketrans('', '', ".,'")
    words = [w.translate(table) for w in words]
    return words

def padding(sequence):
    while len(sequence) < 10:
        sequence.append("<PAD>")
    while len(sequence) > 10:
        sequence = sequence[:-1]
    return sequence

#function to map the list of words to vectors
def map_words_to_vectors (text):
    list_of_vectors = []
    for word in text:
        try:
            if word == "</s>":
                list_of_vectors.append(Start_vector)
            elif word == "<PAD>":
                list_of_vectors.append(Padding_vector)
            else:
                list_of_vectors.append(word_vectors[word])
        except:
            list_of_vectors.append(Unk_vector)
    return list_of_vectors

def one_hot(sentence):
    temp = []
    for number in sentence:
        array = np.zeros(keep_n+2) #add two .. 1) </s> 3000 2)padding 3001
        array[number] = 1
        temp.append(array)
    return temp

def indexing(Corpus,keep_n,length,samples):
    sentences = []
    for i in range(len(Corpus)):
        sentences.append(Corpus['Output'][i])
    dct = Dictionary(sentences)
    dct.filter_extremes(keep_n)
    dictionary = (dct.token2id)
    newsentences = []
    for sentence in sentences:
        newsentence = [keep_n] #pad the sentence with the </s> token
        for item in sentence:
            if (item in dictionary):
                newsentence.append(dictionary[item])
            else:
                pass
        newsentences.append(newsentence)
    X_train = sequence.pad_sequences(newsentences, maxlen=length, value=3001, padding="post", truncating="post")
    temp = []
    for i in range(samples):
        temp.append(one_hot(X_train[i]))
    return temp
#collect all the lists into one list
def collect(df,column):
    output = []
    for i in range(len(df)):
        output.append(df[column][i])
    output = np.array(output)
    return output



def main ():
    Corpus = pd.read_csv(r"corpus.csv",encoding = "ISO-8859-1")
    Corpus['Input'] = Corpus['Input'].astype(str).apply(splitting) #Makes the Input column a list of words
    Corpus['Output'] = Corpus['Output'].astype(str).apply(splitting) #Makes the Output column a list of words
    Corpus['Input'] = Corpus['Input'].apply(padding)
    Corpus['Output'] = Corpus['Output'].apply(padding)
    Corpus['Output_Vectors'] = Corpus['Output'].apply(map_words_to_vectors) #List of word vectors
    Corpus['Input_Vectors'] = Corpus['Input'].apply(map_words_to_vectors)
    one_hot_vectors = indexing(Corpus,keep_n,10,3000)
    Output = collect(Corpus,"Output_Vectors")
    Input = collect(Corpus,"Input_Vectors")
    return one_hot_vectors,Output,Input