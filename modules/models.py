# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:08:41 2023

@author: Fabrice
"""



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout









def CNN_model(input_dim, output_dim, weight, input_length, model_name='model_v1'):
    '''
        par défaut, le nom du modèle est 'model_v1'. L'utilisateur peut donner
        un autre nom au modèle. Parmis les arguments du modèle à spécifier:
            - input_dim. C'est la taille de la matrice. ex: len(embedding_matrix)
            - output_dim: C'est la taille des vecteurs représentant les tokens. ex: word2vec_model.vector_size
            - weight: c'est les poids initiaux du kernel. En général,il correspond 
            aux valeurs de la matrice chargée par les vecteurs. ex: [embedding_matrix]
            - input_lenght: c'est la taille d'une séquence après le padding. ex: maxlen
    '''
    
    model_name = Sequential()
    model_name.add(Embedding(input_dim=input_dim, output_dim=output_dim,
                        weights=weight, input_length=input_length, trainable=False))
    model_name.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model_name.add(GlobalMaxPooling1D())
    model_name.add(Dense(128, activation='relu'))
    model_name.add(Dropout(0.2))
    model_name.add(Dense(1, activation='sigmoid'))
    return model_name



def DNN_model(dimension, model_name='model_v2'):
    
    model_name = Sequential()
    model_name.add(Dense(64, activation='relu', input_shape=(dimension,)))
    model_name.add(Dense(16, activation='relu'))
    model_name.add(Dense(1, activation='sigmoid'))
    return model_name

def DNN_model_multi(dimension, model_name='model_v3'):
    
    model_name = Sequential()
    model_name.add(Dense(128, activation='relu', input_shape=(dimension,)))
    model_name.add(Dropout(0.5))
    model_name.add(Dense(64, activation='relu'))
    model_name.add(Dropout(0.5))
    model_name.add(Dense(32, activation='relu'))
    model_name.add(Dense(11, activation='softmax'))
    return model_name