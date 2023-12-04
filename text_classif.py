# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:52:10 2023

@author: Fabrice
"""
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
#import seaborn as sns
import matplotlib.pyplot as plt
from modules import preprocesser, sampler, models
from sklearn.metrics import confusion_matrix, classification_report
from plot_graphic import plotClassDistribution, plot_classification_report, plot_confusion_matrix


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
#xTrain, xTest, yTrain, yTest = preprocesser.data_splitter_8020(X=df1.avis, y=df1.label1)
xTrain, xVal, xTest, yTrain, yVal, yTest = preprocesser.data_splitter_702010(X=df1.avis, y=df1.label1)


#_______________________________________________________________________________________________________________
## la fonction bar de pyplot trace graphiques en barres. Elle prend en argument les valeurs en abscisse, puis en ordonnée
plt.figure(figsize=(3, 3)) #définit la taille de la figure
plt.bar(['0', '1'],df1.label1.value_counts(normalize=True)) # en abscisse on a les 2 classes (0 et 1), et en ordonnées, on a les effectifs pour chaque classe
plt.ylabel('frequence relative'); plt.xlabel('classes') # renommer les axes
plt.title('répartition des avis par classe') #ajout d'un titre au graphe
plt.show() #afficher le graphique


#______________________________________________________________________________________________________________
#data augmentation
aug_feature, aug_target = sampler.min_ups1(X_train=xTrain, y_train=yTrain)


#______________________________________________________________________________________________________________
'''
Vectorisation avec l'algo word2vec de la bibliothèque gensim'
'''

#Vectoriser le jeu d'entrainement avec l'algorithme word2vec de gensim
#les paramètres de cette fonction sont expliqués dans le module 'preprocesser'
word2vec_model = preprocesser.gensim_w2v(corpus=aug_feature, nb_windows=5, min_count=3, sg=0)

embedding_matrix = preprocesser.matrix_embbeder(model=word2vec_model)

#charger la matrice avec les vecteurs correspondants aux mots du corpus
for word, i in word2vec_model.wv.key_to_index.items():
    embedding_vector = word2vec_model.wv[word]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

features =aug_feature.apply(lambda x: preprocesser.text_to_indices(x, word2vec_model)) #transforme le jeu d'entrainement en listes d'index


# Remplissage des séquences pour qu'elles aient la même longueur
# après vérification la plus longue séquence a 41 mot. Donc on peut 'paddé' à 50 sans risquer de tronquer certaines séquences
maxlen=50
padded_features = pad_sequences(features, maxlen=maxlen)

#mettre en forme les features de validation et de test
val_features = xVal.apply(lambda x: preprocesser.text_to_indices(x, word2vec_model))
padded_val_features = pad_sequences(val_features, maxlen=maxlen)


test_features = xTest.apply(lambda x: preprocesser.text_to_indices(x, word2vec_model))
padded_test_features = pad_sequences(test_features, maxlen=maxlen)


y_train = np.asarray(aug_target).astype('float32')
y_val = np.asarray(yVal).astype('float32')
y_test = np.asarray(yTest).astype('float32')

#_____________________________________________________________________________________________________________________________
# Création du modèle CNN
'''
les détails de l'architecture et l'utilisation de la fonction CNN_model' sont dans le
module models.py
'''
model_nlp=models.CNN_model(input_dim=len(embedding_matrix), output_dim=word2vec_model.vector_size,
                 weight=[embedding_matrix], input_length=maxlen)
#____________________________________________________________________________________
# Compilation du modèle
model_nlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
history=model_nlp.fit(padded_features, y_train, validation_data=(padded_val_features, y_val), epochs=2, batch_size=64)

# Évaluation du modèle
#loss, accuracy = model_nlp.evaluate(padded_test_features, yTest)
#print(f'Accuracy: {accuracy * 100:.2f}%')


predictions= model_nlp.predict(padded_test_features)

# Convertir les probabilités prédites en classes (0 ou 1)
predicted_classes = (predictions > 0.5).astype(int)

# Construire la matrice de confusion
confusion_mat = confusion_matrix(y_test, predicted_classes)

# Afficher la matrice de confusion
print("Matrice de Confusion :\n", confusion_mat)

# Afficher le rapport de classification
print("Rapport de Classification :\n", classification_report(y_test, predicted_classes))

# visualiser l'entrainement du modèle
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


#####################################################################"


plotClassDistribution(data=df1.label1, graphic_dir=graphic_dir, plot_name='distribution_classifBinaire.png')


plot_confusion_matrix(y_true=y_test, y_pred=predicted_classes, graphic_dir=graphic_dir, pcm_name='cf_binaryClassif')

fig_a2, ax_2 = plot_classification_report(y_test, predicted_classes, graphic_dir=graphic_dir,
                    title='Classification Report: classification binaire',
                    figsize=(8, 3), dpi=400,
                    #target_names=None, 
                    save_fig_path = graphic_dir + "/binaryClassif.png")
