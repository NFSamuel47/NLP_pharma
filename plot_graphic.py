# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:55:32 2023

@author: Fabrice
"""

import pandas as pd
import numpy as np
import seaborn as sns
import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def plotClassDistribution(data, graphic_dir, plot_name):
    
    
    # la fonction bar de pyplot trace graphiques en barres. Elle prend en argument les valeurs en abscisse, puis en ordonnée
    unique_values=sorted(data.unique())
    plt.figure(figsize=(8, 4)) #définit la taille de la figure
    plt.ylabel('frequence relative'); plt.xlabel('classes') # renommer les axes
    plt.title('répartition des avis par classe') #ajout d'un titre au graphe
    gph = plt.bar(unique_values, data.value_counts(normalize=True)[unique_values]) # en abscisse on a les differentes classes, et en ordonnées, on a les effectifs pour chaque classe
    plt.xticks(unique_values) #affichage des valeurs sur l'axe des abscisses
    barre = gph[0].figure  #récupérer la figure dans l'objet BarContainer
    #plt.show() #afficher le graphique
    barre.savefig(graphic_dir +'/'+ plot_name)
    return barre


def plot_confusion_matrix (y_true, y_pred, graphic_dir, pcm_name):
    ''' plot and save confusion matrix in the specified directory:
        y_true is the target label; y_pred is predicted label;
        pcm_name is the name of the figure in the directory.
        
    '''
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp = disp.plot(cmap=plt.cm.Blues_r)
    return disp.figure_.savefig(graphic_dir +'/'+ pcm_name)
    
def plot_classification_report(y_test, y_pred, graphic_dir, title='Classification Report', figsize=(8, 3), dpi=400, save_fig_path=None, **kwargs):
    """
    Plot the classification report of sklearn
    
    Parameters
    ----------
    y_test : pandas.Series of shape (n_samples,)
        Targets.
    y_pred : pandas.Series of shape (n_samples,)
        Predictions.
    title : str, default = 'Classification Report'
        Plot title.
    fig_size : tuple, default = (8, 3)
        Size (inches) of the plot.
    dpi : int, default = 70
        Image DPI.
    save_fig_path : str, defaut=None
        Full path where to save the plot. Will generate the folders if they don't exist already.
    **kwargs : attributes of classification_report class of sklearn
    
    Returns
    -------
        fig : Matplotlib.pyplot.Figure
            Figure from matplotlib
        ax : Matplotlib.pyplot.Axe
            Axe object from matplotlib
    """    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
    clf_report = classification_report(y_test, y_pred, output_dict=True, **kwargs)
    keys_to_plot = [key for key in clf_report.keys() if key not in ('accuracy', 'macro avg', 'weighted avg')]
    df = pd.DataFrame(clf_report, columns=keys_to_plot).T
    #the following line ensures that dataframe are sorted from the majority classes to the minority classes
    df.sort_values(by=['precision'], inplace=True) 
    
    #first, let's plot the heatmap by masking the 'support' column
    rows, cols = df.shape
    mask = np.zeros(df.shape)
    mask[:,cols-1] = True
 
    ax = sns.heatmap(df, mask=mask, annot=True, cmap="mako", fmt='.3g',
            vmin=0.0,
            vmax=1.0,
            linewidths=2, linecolor='white'
                    )
    
    #then, let's add the support column by normalizing the colors in this column
    mask = np.zeros(df.shape)
    mask[:,:cols-1] = True    
    
    ax = sns.heatmap(df, mask=mask, annot=True, cmap="mako", cbar=False,
            linewidths=2, linecolor='white', fmt='.0f',
            vmin=df['support'].min(),
            vmax=df['support'].sum(),         
            norm=mpl.colors.Normalize(vmin=df['support'].min(),
                                      vmax=df['support'].sum())
                    ) 
            
    plt.title(title)
    plt.xticks(rotation = 0)
    plt.yticks(rotation = 360)
         
    if (save_fig_path != None):
        path = pathlib.Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path)
    
    return fig, ax






