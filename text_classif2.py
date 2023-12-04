# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:15:58 2023

@author: Fabrice
"""

import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
from plot_graphic import plotClassDistribution, plot_classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from modules import preprocesser, sampler, models
from modules.preprocesser import data_splitter_702010, OneHotEncoder, makeDico
from modules.models import DNN_model


#Creation of directory where graphics will be saved
graphic_dir = "graphics"
if not os.path.exists(graphic_dir):
    os.makedirs(graphic_dir)
    print(f"Directory '{graphic_dir}' created.")
else:
    print(f"Directory '{graphic_dir}' already exists.")


df = pd.read_csv('data_defi3.csv', sep=";")  #importer les données dans la variable df

# créer un dictionnaire pour associer des valeurs présentes dans la colonne PLT à des étiquettes d'interet (0 et 1)
dic1={4:1, 5:1, 6.3:1, 6.4:1, 1.1:0, 5.3:1, 4.1:1, 3.1:0, 10.0:0, 1.2:0, 5.1:1, 8.5:0, 2.2:0, 11.0:0, 8.4:0, 6.2:0, 4.2:1, 9.1:0, 8.1:0, 8.3:0, 6.1:0, 3.2:0, 2.4:0, 8.2:0,5.2:1,1.3:0, 2.1:0, 7.0:0}

# création d'une liste de mots jugés non pertinents pour la compréhension du contexte, malgré le premier prétraitement.
no_words=['mg','au','cp','eb', 'ab','kg','aux','de','est','le','la','les','sur','en','dans','pour','des','du','ml','cas','il','elle','et','si','ou', 'ce']


df1 = preprocesser.formatted_table(table=df, dictionary=dic1, colnames=['avis', 'label1'])
df1.avis = preprocesser.formatted_sequences(corpus=df1.avis, nowords_list=no_words)

#______________________________________________________________________________________________________________
# créer le dictionaire des mots de mon corpus
dico = makeDico(df1.avis)
# Transformer chaque document tokenisé en séquence d'indices
df1['indices'] = df1['avis'].apply(lambda x: dico.doc2idx(x))


xTrain, xVal, xTest, yTrain, yVal, yTest = data_splitter_702010(X=df1.indices, y=df1.label1)

#______________________________________________________________________________________________________________
'''
Vectorisation avec la méthode du one hot encoding'
'''

x_vec_train = OneHotEncoder(xTrain)
x_vec_val = OneHotEncoder(xVal)
x_vec_test = OneHotEncoder(xTest)

y_train = np.asarray(yTrain).astype('float32')
y_val = np.asarray(yVal).astype('float32')
y_test = np.asarray(yTest).astype('float32')


#_____________________________________________________________________________________________________________________________
# Création du modèle DNN
'''
les détails de l'architecture et l'utilisation de la fonction DNN_model' sont dans le
module models.py
'''

model2_nlp = DNN_model(dimension=10000)

#____________________________________________________________________________________
# Compilation du modèle
model2_nlp.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Entraînement du modèle
history=model2_nlp.fit(x_vec_train, y_train, validation_data=(x_vec_val, y_val), epochs=4, batch_size=128)

#____________________________________________________________________________________
predictions1= model2_nlp.predict(x_vec_test)

# Convertir les probabilités prédites en classes (0 ou 1)
predicted_classes = (predictions1 > 0.55).astype(int)

# Construire la matrice de confusion
confusion_mat = confusion_matrix(y_test, predicted_classes)

# Afficher la matrice de confusion
print("Matrice de Confusion classification binaire:\n", confusion_mat)

# Afficher le rapport de classification
print("Rapport de Classification binaire:\n", classification_report(y_test, predicted_classes))


#____________________________________________________________________________________
plotClassDistribution(data=df1.label1, graphic_dir=graphic_dir, plot_name='distribution_classes.png')


plot_confusion_matrix(y_true=y_test, y_pred=predicted_classes, graphic_dir=graphic_dir, pcm_name='cf_binaryClassif')

fig_a2, ax_2 = plot_classification_report(y_test, predicted_classes, graphic_dir=graphic_dir,
                    title='Classification Report: classification binaire',
                    figsize=(8, 3), dpi=400,
                    #target_names=None, 
                    save_fig_path = graphic_dir + "/clr_binaryClassif.png")

