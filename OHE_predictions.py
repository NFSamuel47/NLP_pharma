# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:55:11 2023

@author: Fabrice
"""

import os
import pandas as pd
import numpy as np
import pickle #pour charger les fichiers au format .pkl (dictionnaire du corpus sur lequel le model a été entrainé)
from keras.models import load_model #pour charger le modèle entrainé
from modules.preprocesser import formatted_test_set, formatted_sequences, OneHotEncoder



models_dir = "modeles"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Directory '{models_dir}' created.")
else:
    print(f"Directory '{models_dir}' already exists.")
    
    

data = pd.read_csv("valid_set.csv", sep=';')

print("le nombre de documents manquants dans le set de test est de: ", data['Avis.Pharmaceutique'].isnull().sum())

# création d'une liste de mots jugés non pertinents pour la compréhension du contexte, malgré le premier prétraitement.
no_words=['mg','au','cp','eb', 'ab','kg','aux','de','est','le','la','les','sur','en','dans','pour','des','du','ml','cas','il','elle','et','si','ou', 'ce']

dataset = formatted_test_set(data)
dataset.avis = formatted_sequences(corpus=dataset.avis, nowords_list=no_words)

print("Chargement du dictionnaire de mots du set d'entrainement")
with open(models_dir + '/dico.pkl', 'rb') as fichier:
    dictionnaire = pickle.load(fichier)


# Transformer chaque document tokenisé en séquence d'indices
dataset['indices'] = dataset['avis'].apply(lambda x: dictionnaire.doc2idx(x))

#______________________________________________________________________________________________________________
'''
Vectorisation avec la méthode du one hot encoding'
'''
vec_data = OneHotEncoder(dataset['indices'])

print(" ------------------->  preprocessing terminé")
print ("_______________________________________________________________________")


#charger le modèle de prediction multiple sauvegardé après entrainement     
cm_model = load_model(models_dir + "/multiclassif1.h5")

print("Modèle multiclasse chargé avec succès")

#____________________________________________________________________________________
cm_prediction= cm_model.predict(vec_data)

# Convertir les probabilités prédites en classes de sortie
predicted_classes1 = np.argmax(cm_prediction, axis=1)

print("Prédictions multiclasses terminées")

# Créer un DataFrame pandas avec les classes prédites
df_prediction_cm = pd.DataFrame({'predicted_label': predicted_classes1})

# Exporter le DataFrame vers un fichier CSV
df_prediction_cm.to_csv('./prediction_multi.csv', index=True, sep = ";")

print("Fichier des prédictions pour classification multiple exporté au format .csv")

print ("_______________________________________________________________________")


#charger le modèle de classification binaire sauvegardé après entrainement     
cb_model = load_model(models_dir + "/binaryclassif1.h5")

print("Modèle binaire entrainé sur données vectorisées par OneHotEncoding chargé avec succès")

#____________________________________________________________________________________
cb_prediction= cb_model.predict(vec_data)

# Convertir les probabilités prédites en classes (0 ou 1)
predicted_classes2 = (cb_prediction > 0.55).astype(int)

print("Prédictions classification binaire terminées")

# Créer un DataFrame pandas avec les classes prédites
indices = dataset.index #récupérer les index du dataFrame après suppression des NA


df_prediction_cb = pd.DataFrame({'predicted_label': predicted_classes2.flatten()}, index=indices)

# Exporter le DataFrame vers un fichier CSV
df_prediction_cb.to_csv('./prediction_binaire.csv', index=True, sep = ';')

print("Fichier des prédictions pour classification binaire exporté au format .csv")

