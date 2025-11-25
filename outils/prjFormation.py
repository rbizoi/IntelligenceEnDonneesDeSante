import pandas as pd, numpy as np, seaborn as sns, warnings, os, sys, time, copy as cp, pickle

from datetime import datetime as dt
from matplotlib import pyplot as plt

import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects

import plotly.express as px
import plotly.graph_objects as go

from scipy.cluster.hierarchy import dendrogram, linkage

palette = [ "#030aa7", "#e50000", "#fe9929", "#005f6a", "#6b7c85", "#751973", 
            "#0485d1", "#ff7855", "#fbeeac", "#0cb577", "#95a3a6", "#c071fe", 
            "#92c5de", "#f4b182", "#feffba", "#18d17b", "#c5c9c7" ,"#caa0ff",
            "#d1e5f0", "#fddbc7", "#ffffcb", "#20f1a3", "#e8ece6", "#dfc5fe", 
            "#030aa7", "#e50000", "#fe9929", "#005f6a", "#6b7c85", "#751973", 
            "#0485d1", "#ff7855", "#fbeeac", "#0cb577", "#95a3a6", "#c071fe", 
            "#92c5de", "#f4b182", "#feffba", "#18d17b", "#c5c9c7" ,"#caa0ff",
            "#d1e5f0", "#fddbc7", "#ffffcb", "#20f1a3", "#e8ece6", "#dfc5fe", 
            "#030aa7", "#e50000", "#fe9929", "#005f6a", "#6b7c85", "#751973", 
            "#0485d1", "#ff7855", "#fbeeac", "#0cb577", "#95a3a6", "#c071fe", 
            "#92c5de", "#f4b182", "#feffba", "#18d17b", "#c5c9c7" ,"#caa0ff",
            "#d1e5f0", "#fddbc7", "#ffffcb", "#20f1a3", "#e8ece6", "#dfc5fe",     
            ]
            
outlier = "#fe01b1"

def controleExistenceRepertoire( repertoire, create_if_needed=True):
    """Voir si le répertoire existe. S'il n'existe pas il est créé."""
    path_exists = os.path.exists(repertoire)
    if path_exists:
        if not os.path.isdir(repertoire):
            raise Exception("Trouvé le nom  "+repertoire +" mais c'est un fichier, pas un répertoire")
            # return False
        return True
    if create_if_needed:
        os.makedirs(repertoire)


class initProjet():
    def __init__(self, 
                 repertoireRacine = '.',
                 nomProjet        = 'dummy'):
        self.repertoireRacine  = repertoireRacine 
        self.nomProjet         = nomProjet        
        self.repertoireProjet  = os.path.join(self.repertoireRacine, self.nomProjet)
        self.repertoireDonnees = os.path.join(self.repertoireProjet, 'repertoire.donnees')
        self.repertoireImages  = os.path.join(self.repertoireProjet, 'repertoire.images')        
        controleExistenceRepertoire(self.repertoireProjet);
        controleExistenceRepertoire(self.repertoireDonnees);
        controleExistenceRepertoire(self.repertoireImages);
                
    def sauvegarderImage( self, fichier, dpi=100,bbox_inches='tight'):
        """Enregistrez la figure. Appelez la méthode juste avant plt.show ()."""
        controleExistenceRepertoire(self.repertoireImages)
        plt.savefig(os.path.join(self.repertoireImages,
                                 fichier+f"--{dt.now().strftime('%Y_%m_%d_%H.%M.%S')}.png"), 
                                 dpi=dpi, 
                                 #bbox_inches='tight'
                                 )

    def sauvegarderImageSNS( self, sns_plot, fichier):
        """Enregistrez la figure. Appelez la méthode juste avant plt.show ()."""
        controleExistenceRepertoire(self.repertoireImages)
        fig = sns_plot.get_figure()
        fig.savefig(os.path.join(self.repertoireImages,fichier+'.png'))




def formatPct(pct, allvals):
    total = int(round(pct/100. * np.sum(allvals)))
    return "{:.2f}%\n({:d})".format(pct, total)  

def affichageDistribution(colonne, couleur, ax, nom=''):
    graph = sns.distplot(colonne, color=couleur, ax=ax)
    graph.set(ylabel=None)
    moyenne, mediane = float(colonne.mean()), \
                   float(colonne.median())
    
    ax.axvline(moyenne, color='g', linestyle='-', label=f"{nom:12s} mean   = {moyenne:0.4f}", lw=2)
    ax.axvline(mediane, color='b', linestyle='--', label=f"{nom:12s} median = {mediane:0.4f}", lw=2)
    graph.legend(loc="upper right")
    
def add_median_labels(ax, precision='.1f'):
    lines = ax.get_lines()
    # determine number of lines per box (this varies with/without fliers)
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    # iterate over median lines
    for median in lines[4:len(lines):lines_per_box]:
        # display median value at center of median line
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1]-median.get_xdata()[0]) == 0 else y
        text = ax.text(
                       x, 
                       y, 
                       f'{value:{precision}}', 
                       verticalalignment='center',
                       horizontalalignment='center', 
                       fontweight='bold', 
                       color='black',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.6),
                      )
        # créer une bordure de couleur médiane autour du texte blanc pour le contraste 
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])  

def afficheColonneCible(affiche:pd.DataFrame, colonne:str, cible:str, clouleurs:list, title:str):
    plt.figure(figsize=(24,12))
    graph = sns.countplot(x=colonne,
                          hue=cible,
                          data=affiche.sort_values([colonne,cible]),
                          palette=clouleurs);
    hauteur = affiche.groupby([cible])[colonne].count().mean()/20
    for patche in graph.patches:
        if patche.get_height() > 0 :
            graph.text(
                        patche.get_x() + patche.get_width() / 2 ,
                        5 if hauteur < 5 else hauteur,
                        #10 if int(patche.get_height()*0.3) < 10 else int(patche.get_height()*0.3),
                        int(patche.get_height()),
                        color='black',
                        rotation='vertical',
        #                 size='large',
        #                 fontsize='large',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6),
                        verticalalignment='center',
                        horizontalalignment='center',
                       )       
    
    graph.set_ylabel('occurrences');
    graph.set_xlabel('');
    graph.set_title(title, y=1.05, size=36)
    # graph.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0);            

def createLinkageMatrix(model):

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    return linkage_matrix   

def afficheDendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Classification Hiérarchique Ascendante')
        plt.xlabel('Villes ou (taille du cluster)')
        plt.ylabel('Distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def affichageDonnees2d(donnees, palette, hue=None, afficheCentres = False, afficheTicks = True, tailePoint=200, alpha=0.8, tailleImage=12, ax=None, nomDonnees=None):
    colonnes = donnees.columns
    cible    = donnees.index.name if hue is None else hue
    if ax is None :
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(tailleImage,tailleImage))

    sns.scatterplot(x=colonnes[0],
                    y=colonnes[1], 
                    hue=cible, 
                    data=donnees.reset_index(),
                    alpha=alpha,       
                    s=tailePoint,    
                    palette=palette,
                    legend=None,
                    ax=ax) ;  
    if afficheCentres : 
        affichage = donnees.reset_index().copy()
        centres = affichage.groupby(cible).mean().reset_index().rename(columns={cible:'Classe'})
        sns.scatterplot(x=colonnes[0],
                        y=colonnes[1], 
                        # hue='Classe', 
                        data=centres,
                        alpha=alpha,       
                        s=tailePoint*3,    
                        # palette=palette,
                        c="#d8dcd6",
                        ax=ax) ; 
        
        for i,x,y in zip(centres.Classe,centres[colonnes[0]],centres[colonnes[1]]) : 
            ax.scatter(x, y, marker=f'${i}$', alpha=alpha, s=tailePoint, c="#d8dcd6", edgecolor="#030aa7" )     

    if not afficheTicks : 
        ax.set_xlabel('');ax.set_ylabel('');ax.set_xticks(());ax.set_yticks(())   

    if nomDonnees is not None :    
        ax.set_title(nomDonnees,fontweight='bold',fontsize='x-large')


def affichageDonnees3d(donnees, palette):
    colonnes = donnees.columns
    cible    = donnees.index.name    
    layout = go.Layout({"showlegend": False})
    fig = px.scatter_3d(donnees.reset_index().sort_values(cible), 
                        x=colonnes[0],
                        y=colonnes[1], 
                        z=colonnes[2], 
                        color=cible,
                        size= donnees.reset_index().sort_values(cible)[cible].astype('int32'),
                        # symbol='cible',
                        # text='cible',
                        width=1024,
                        height=1024,
                        color_discrete_sequence=palette
                       )
    fig.show()

def createColumnsQualitatives(donnees,colonne):
    valeurs = donnees[colonne].sort_values().unique()
    for i in valeurs :
        nom = colonne+'='+str(i)
        donnees[nom] = donnees[colonne].apply(lambda x : 1 if x==i else 0)

    donnees.drop(labels=colonne, axis=1, inplace=True)
    return donnees