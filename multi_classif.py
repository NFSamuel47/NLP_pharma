# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:02:02 2023

@author: Fabrice
"""
import os
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from plot_graphic import plotClassDistribution, plot_classification_report, plot_confusion_matrix
from modules import preprocesser
from modules.preprocesser import float2int, makeDico, data_splitter_702010, OneHotEncoder
from modules.models import DNN_model_multi
from keras.callbacks import EarlyStopping


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

# recodage de la colonne PLT dans une colonne nommée label2 à onze classes
df['label2'] = df['PLT'].apply(lambda x: float2int(x))

df1 = preprocesser.formatted_table(table=df, dictionary=dic1, colnames=['avis', 'label2', 'label1'])
df1.avis = preprocesser.formatted_sequences(corpus=df1.avis, nowords_list=no_words)

#______________________________________________________________________________________________________________
# créer le dictionaire des mots de mon corpus
dico = makeDico(df1.avis)
# Transformer chaque document tokenisé en séquence d'indices
df1['indices'] = df1['avis'].apply(lambda x: dico.doc2idx(x))


xTrain, xVal, xTest, yTrain, yVal, yTest = data_splitter_702010(X=df1.indices, y=(df1.label2-1))

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

model3_nlp = DNN_model_multi(dimension=10000)

#____________________________________________________________________________________
# Compilation du modèle
#on utilise sparse_categorical_crossentropy quand les labels sont des entiers
#s'ils sont des flottants, on utilisera categorical_crossentropy

# definition des poids à associer aux classes
classe_weights = {0.0: 0.0005089113306923786, 3.0: 0.0018705035188649405, 7.0: 0.0019455769378981169, 10.0: 0.0036312752628138477, 2.0: 0.005916842649422226, 5.0: 0.005934296123829108, 4.0: 0.006889473483661151, 1.0: 0.007577124879706591, 9.0: 0.007767283063325879, 8.0: 0.28738839187510734, 6.0: 0.6705703208746785}

model3_nlp.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Définition de l'arrêt anticipé
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# Entraînement du modèle
history=model3_nlp.fit(
    x_vec_train, y_train, 
    validation_data=(x_vec_val, y_val),
    epochs=10, batch_size=64,
    callbacks=[early_stopping],
    class_weight=classe_weights)

#
#____________________________________________________________________________________
predictions2= model3_nlp.predict(x_vec_test)

# Convertir les probabilités prédites en classes de sortie
predicted_classes = np.argmax(predictions2, axis=1)

# Construire la matrice de confusion
confusion_mat = confusion_matrix(y_test, predicted_classes)

# Afficher la matrice de confusion
print("Matrice de Confusion :\n", confusion_mat)

# Afficher le rapport de classification
print("Rapport de Classification :\n", classification_report(y_test, predicted_classes))


#____________________________________________________________________________________
plotClassDistribution(data=df1.label2, graphic_dir=graphic_dir, plot_name='distribution_multiclasse.png')


plot_confusion_matrix(y_true=y_test, y_pred=predicted_classes, graphic_dir=graphic_dir, pcm_name='cf_multiClassif')

fig_a2, ax_2 = plot_classification_report(y_test, predicted_classes, graphic_dir=graphic_dir,
                    title='Classification Report: classification multiple',
                    figsize=(8, 3), dpi=400,
                    #target_names=None, 
                    save_fig_path = graphic_dir + "/clr_multiClassif.png")
