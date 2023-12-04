# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:20:23 2023

@author: Fabrice
"""

import pandas as pd
from sklearn.utils import shuffle
#charger le module de gensim qui gère les dictionnaires préentrainés
from gensim.models import KeyedVectors 




def min_ups1(X_train, y_train):
    
    '''
    L'une des méthodes de gestion de déséquilibre des classes est 
    l'augmentation des données de synthèse dans la classe minoritaire. Seulement,
    dans le cas du TALN, il est difficile d'utiliser des algorithmes de SMOTE,
    car adaptés aux features numériques. L'idée principale étant l'ajout de nouvelles
    données sans modifier la sémantique, on peut procéder par substitution d'un mot
    par son synonyme dans une séquence de mot. Pour ce faire, il faut avoir à sa
    disposition un vocabulaire pour ces mots. La bibliothèque Gensim permet de charger
    des vocabulaires construits à partir des modèles préentrainés.
    Le modules KeyedVectors avec un fichier téléchargé et compatible avec
    le corpus en francais peut fournir un vocabulaire assez riche pour effectuer
    cette tache d'augmentation des données
    '''
    
    #jointure des X et y du jeu d'entrainement dans une dataframe temporaire pour gérer le déséquilibre des classes.
    data = pd.DataFrame(zip(X_train, y_train))
    data.columns = ['X_train', 'y_train']


    # Diviser les données par classe
    minority_train = data[data["y_train"] == 1]["X_train"]
    majority_train = data[data["y_train"] == 0]["X_train"]

    
    
    path1 = "./modules/frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin"
    model1 = KeyedVectors.load_word2vec_format(path1, binary=True, unicode_errors="ignore")
    
    def replace_with_synonym(word, modele=model1):
        try:
            synonym = modele.most_similar(word, topn=1)[0][0]
            return synonym
        except KeyError:
            return word

    # Appliquer la data augmentation sur la classe minoritaire
    aug_min_train = []

    for sequence in minority_train:
        augmented_sequence = [replace_with_synonym(word) for word in sequence]
        aug_min_train.append(augmented_sequence) #aug_min_train est une liste de listes, et non une dataframe comme monirity_train

    # Concaténer les données augmentées avec la classe minoritaire
    new_min_train = pd.concat([pd.Series(aug_min_train), minority_train], ignore_index=True)

    #créer une bdd temporaire contenant les X et y de la classe minoritaire augmentée
    min_data = pd.DataFrame(zip(new_min_train, pd.Series(len(new_min_train)*[1])))

    #construire aussi une bdd temporaire pour les X et y de la classe majoritaire
    max_data = pd.DataFrame(zip(majority_train, pd.Series(len(majority_train)*[0])))

    #concatener les deux bdd temporaires pour faire le jeu d'entrainement avec la classe minoritaire augmentée
    final_train_data=pd.concat([min_data, max_data], ignore_index=True)
    final_train_data.columns = ['X_train', 'y_train']

    #Melanger les donnees dans le jeu d'entrainement, puis diviser en X et Y
    train_features, train_targets = shuffle (final_train_data.X_train, final_train_data.y_train, random_state = 123)
    return train_features, train_targets