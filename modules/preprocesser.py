# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:50:25 2023

@author: Fabrice
"""
import spacy
import numpy as np
import gensim #cette bibliothèque contient des modules pour le prétraitement (tokenisation, vectorisation, ...)
from sklearn.model_selection import train_test_split  #separation des données en entrainement et en test

def formatted_table(table, dictionary, colnames):
    '''
    cette fonction récupère une dataframe ayant les colonnes 'Libellé.Prescription',
    'Avis.Pharmaceutique' et 'PLT'. Ensuite selon le dictionnaire passé en entré,
    retourne une autre dataframe contenant des avis pharmaceutiques classés dans la
    colonne 'avis', en fonction de la colonne 'label'.
    Le dictionnaire permet de redéfinir les classes initialement présentes dans PLT.
    '''
    
    #appliquer ce dictionnaire à la colonne PLT pour transformer les valeurs présentes en etiquettes d'interet
    dic_label=table.PLT.apply(lambda x: dictionary[x])
    
    #ajouter à df, la colonne contenant les étiquettes obtenues précédemment
    table=table.assign(label=dic_label)
    
    #supprimer la colonne (le champs) intitulée 'Libellé.Prescription' car elle n'est pas utilisée
    table1=table.drop('Libellé.Prescription',axis=1)
    table1=table1.drop('PLT',axis=1) #la colonne PLT ne sera plus utile, car remplacée par la colonne "label"
    
    # renommer les colonnes
    table1.columns = colnames
    
    #supprimer les lignes complètent qui contiennent des valeurs manquantes dans n'importe quelle de leurs colonnes.
    table1=table1.dropna()
    
    return table1


def formatted_test_set(table, colnames=['avis']):
    '''
    cette fonction récupère une dataframe ayant les colonnes 'Libellé.Prescription' et
    'Avis.Pharmaceutique' et retourne une autre dataframe contenant des avis 
    pharmaceutiques dans une colonne renommée 'avis. Les NA sont supprimés'
    '''
    
    #supprimer la colonne (le champs) intitulée 'Libellé.Prescription' car elle n'est pas utilisée
    table1=table.drop('Libellé.Prescription',axis=1)
    
    # renommer les colonnes
    table1.columns = colnames
    
    #supprimer les lignes complètent qui contiennent des valeurs manquantes dans n'importe quelle de leurs colonnes.
    table1=table1.dropna()
    
    return table1


def formatted_sequences(corpus, nowords_list):
    '''Cette fonction utilise la bibliothèque gensim pour prétraiter la colonne 'avis'
        contenant du texte. Parmis les opérations de prétraitrement, on a la 
        transformation des textes en séquences, la tokénisation des mots, la conversion
        de tous les mots en majuscule. Le préproceseur de Gensim etant spécialisé pour 
        les corpus en anglais, la fonction ajoute aussi une opération de suppression des
        mots vides de sens 'stop worlds', qu'il faudra définir.        
    '''
    
    #texte (sequences) tokénisé, et débarassé des stopwords (en anglais) et transformés en minuscule
    corpus1=corpus.apply(gensim.utils.simple_preprocess)
    
    #suppression des mots du corpus presents dans la nowords_list
    def filter_words(word_list):
        return [word for word in word_list if word not in nowords_list]
    
    corpus2 = corpus1.apply(filter_words)
    return corpus2

#=======================================================================================
nlp = spacy.load("fr_core_news_sm")
# Obtenir la liste des stopwords en français 
stopwords = spacy.lang.fr.stop_words.STOP_WORDS

def formatted_sequences_Spacy(corpus, nowords_list):
    '''Cette fonction est une version légèrement différente de 'formatted sequences'. 
        Ici, on utilise aussi la bibliothèque SpaCy qui possède un vocabulaire qui 
        permet de réaliser la lemmatisation, et de mieux traiter les stopworlds.
    '''
    def text_cleaner(corpus):
        doc = nlp(corpus)
        lemmatized_text = " ".join([token.lemma_ for token in doc])
        # Filtrer les tokens qui ne sont pas des stopwords
        doc2 = nlp(lemmatized_text)
        tokens_without_stopwords = [token.text for token in doc2 if token.text.lower() not in stopwords]
    
        # Rejoindre les tokens pour former le texte nettoyé
        cleaned_text = ' '.join(tokens_without_stopwords)
        return (cleaned_text)

    corpus1=corpus.apply(text_cleaner)

    #texte (sequences) tokénisé, et débarassé des stopwords (en anglais) et transformés en minuscule
    corpus2=corpus1.apply(gensim.utils.simple_preprocess)
    
    #suppression supllémentaire des mots du corpus presents dans la nowords_list
    def filter_words(word_list):
        return [word for word in word_list if word not in nowords_list]
    
    corpus3 = corpus2.apply(filter_words)
    return corpus3

def float2int(value):
    '''
    la fonction récupère la valeure entière d'un nombre décimal.
    le résultat est toujours flottant.
    '''
    #value2 = parseInt(value) "Cette méthode semble un peu moins rapide. fonctionne aussi sur des str
    value2 = value - value%1
    return value2
    


def data_splitter_8020(X, y):
    '''
    divise les données en 80% du jeu d'entrainement et 20% pour l'évaluation
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104,test_size=0.20, shuffle=True, stratify=y)
    return X_train, X_test, y_train, y_test

def data_splitter_702010(X, y):
    '''
    divise les données en 70% du jeu d'entrainement, 20% de validation
    et 10% pour le test.
    
    Etant donné que la fonction 'train_test_split' ne peut pas faire directement
    la séparation en 3 sous-ensembles, on va d'abord faire une première division
    en 90 (trainVal) et 10(test), puis faire une seconde division des 90 en 
    70 pour le train, et 20 pour la validation.
    '''
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, random_state=104,
                                                                test_size=0.10, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, random_state=104,
                                                      test_size=0.22, shuffle=True, stratify=y_train_val)
    return X_train, X_val, X_test, y_train, y_val, y_test


def makeDico(text):
    
    '''
    Cette fonction créé un dictionaire à partir d'un corpus, avec la méthode
    Dictionary de gensim
    '''
    my_dico = gensim.corpora.Dictionary(text)
    return my_dico
    


def gensim_w2v(corpus, nb_windows, min_count, sg):
    
    '''
    Cette fonction fait la vectorisation du corpus avec l'algorithme word2vec du module
    gensim.models
    - nb_windows est le nombre de mots nécessaires pour comprendre le contexte d'un mot;
    - min_count est le nombre d'occurence minimal nécessaire pour  prendre en compte 
    un mot du texte dans le dictionnaire;
    - sg est la stratégie de sémantisation. On peut lui donner les valeur 0 ou 1
    '''
    #paramétrer le modèle de vectorisation
    w2v_model=gensim.models.Word2Vec(window=nb_windows, min_count=min_count, workers=4, sg=sg)

    #construire le vocabulaire à partir du corpus des données d'entrainement
    w2v_model.build_vocab(corpus, progress_per=1000)


    #construire le dictionnaire en entrainant le modèle paramétré sur le vocabulaire du corpus
    w2v_model.train(corpus, total_examples=w2v_model.corpus_count, epochs=20)

    # Sauvegarder le modele de vectorisation ainsi entrainé.
    w2v_model.save("./word2vec.model")
    
    # charger le model entrainé
    model = gensim.models.Word2Vec.load("./word2vec.model")
    return model


def OneHotEncoder(sequences, dimension=10000):
    
    '''
    Cette fonction réalise la vectorisation des séquences, avec la méthode du One Hot
    encoding. Il faut préciser que ce type de vectorisation ne prend pas en compte
    le contexte des mots dans les phrases, mais juste leur présence ou non.
    '''
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results



def matrix_embbeder(model):
    
    '''
    cette matrice contiendra tous les mots du dictionnaire, où chaque mot est représenté
    par un vecteur de taille (100,).   

    '''
    
    # Utilisez .key_to_index pour obtenir la taille du vocabulaire
    vocab_size = len(model.wv.key_to_index) 

    #creer une matrice vide (remplie de zero)    
    embedding_matrix = np.zeros((vocab_size + 1, model.vector_size)) 
    
    return embedding_matrix



def text_to_indices(text, model):
    '''
    convertir les sequences de mots en sequences d'indices
    '''
    
    return [model.wv.key_to_index[word] if word in model.wv else 0 for word in text]
