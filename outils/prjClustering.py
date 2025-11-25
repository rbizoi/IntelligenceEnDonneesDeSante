import pandas as pd, numpy as np, math, seaborn as sns, warnings, os, sys, time, copy as cp, pickle

from datetime import datetime as dt
from matplotlib import pyplot as plt

import matplotlib.font_manager as fm
import plotly.express as px

from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score 

sys.path.append(os.path.abspath('../outils/'))
from prjFormation import palette, outlier

def afficheKMeansClusters(donnees, infoKMeans, cluster=6, ax=None):
    colonnes = donnees.columns
    affichage = donnees.copy()
    affichage['Classe'] = infoKMeans[infoKMeans.clusters==cluster].classes.values[0]
    centres = pd.DataFrame(infoKMeans.loc[infoKMeans.clusters==cluster].centres.values[0],columns=colonnes).reset_index().rename(columns={'index':'Classe'})
    if ax is None :
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,12))
    
    sns.scatterplot(x=colonnes[0],
                    y=colonnes[1], 
                    hue='Classe', 
                    data=affichage,
                    alpha=0.6,       
                    s=200,    
                    palette=palette,
                    legend=None,
                    ax=ax) ; 
    
    sns.scatterplot(x=colonnes[0],
                    y=colonnes[1], 
                    # hue='Classe', 
                    data=centres,
                    alpha=0.9,       
                    s=600,    
                    # palette=palette,
                    c="#d8dcd6",
                    ax=ax) ; 
    
    for i,x,y in zip(centres.Classe,centres[colonnes[0]],centres[colonnes[1]]) : 
        ax.scatter(x, y, 
                   marker=f'${i}$', 
                   alpha=0.9, 
                   s=200, 
                   c="#030aa7",
                   edgecolor="#030aa7")    

    ax.text( affichage[colonnes[0]].max() * 0.85, 
             affichage[colonnes[1]].max() * 0.9, 
             f"{centres.Classe.count():d}",
             color="#030aa7",   
             fontweight='bold', 
             fontsize='x-large',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),);

    return affichage, centres

def affichageClusters(donnees, infoKMeans, nbGraphsLigne=5, tailleImage=12):
    clusters,_ = infoKMeans.shape
    fig,axes = plt.subplots(nrows=clusters//nbGraphsLigne,
                            ncols=nbGraphsLigne,
                            figsize=(tailleImage*nbGraphsLigne,
                                     tailleImage*clusters//nbGraphsLigne))
    
    for i in range(clusters):
        _, _ = afficheKMeansClusters(donnees, infoKMeans, cluster=2+i, ax=axes[i//nbGraphsLigne,i%nbGraphsLigne])

def affichageChoixNombreClusters(infoKMeans, metriques=['wss','silhouette','calinski_harabasz','davies_bouldin','bic'], tailleImage=12, complementNom=None):
    fig,ax = plt.subplots(nrows=1,ncols=len(metriques),figsize=(tailleImage*len(metriques),12))
    for i, metrique in enumerate(metriques):
        sns.scatterplot(x         = 'clusters',
                        y         = metrique,
                        data      = infoKMeans,
                        s         = 200,
                        color     = "#e50000",
                        ax        = ax[i]);
        
        sns.lineplot(   x         = 'clusters',
                        y         = metrique,
                        data      = infoKMeans,
                        estimator = None, 
                        lw        = 2, 
                        color     = "#030764",
                        ax        = ax[i]);
        
        valeurOptimum,nomGraph = 0,'' 
        if metrique == 'davies_bouldin':
            valeurOptimum = infoKMeans.loc[infoKMeans[metrique] == infoKMeans[metrique].min(),'clusters'].values[0]
            nomGraph = f'{metrique}-min'
        elif metrique == 'wss':
            valeurOptimum = KneeLocator(infoKMeans.clusters.values,infoKMeans.wss.values,curve='convex',direction='decreasing').knee
            nomGraph = f'{metrique}-coude'
        else:
            valeurOptimum = infoKMeans.loc[infoKMeans[metrique] == infoKMeans[metrique].max(),'clusters'].values[0]
            nomGraph = f'{metrique}-max'

        nomGraph = f'{nomGraph}\n{complementNom}' if complementNom is not None else nomGraph
            
        if valeurOptimum is not None :    
            ax[i].axvline(valeurOptimum, color="#751973", linestyle='--', label=f"clusters={valeurOptimum:d}", lw=2)        
        
        ax[i].set_xlabel('Clusters');
        ax[i].set_ylabel('');
        ax[i].set_xscale('linear')
        ax[i].legend(loc="upper right")
        ax[i].set_title(nomGraph)

def retrouveDFCluster(infoKMeans, nrCluster = 2):
    if nrCluster < 2 : nrCluster = 2
    if nrCluster > infoKMeans.clusters.max() : nrCluster = infoKMeans.clusters.max()
    d = infoKMeans[['classes','silhouettes']].to_dict()
    return pd.DataFrame({'classes':d['classes'][nrCluster-2],'silhouettes':d['silhouettes'][nrCluster-2]})            

def afficheGraphiqueSilhouettes(donnees, ax, tailleImage=12):
    if ax is None :
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(tailleImage,tailleImage))

    n_clusters = len(donnees.classes.unique())

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        silhouettesCluster = donnees.loc[donnees.classes == i,'silhouettes'].values

        silhouettesCluster.sort()

        tailleCluster = silhouettesCluster.shape[0]
        y_upper = y_lower + tailleCluster       
        
        ax.fill_betweenx( np.arange(y_lower, y_upper),
                          0, 
                          silhouettesCluster,
                          facecolor=palette[i], 
                          edgecolor=palette[i], 
                          alpha=0.5)

        moyenne = donnees.loc[donnees.classes == i,'silhouettes'].mean()
        ax.axvline( x=moyenne, 
                    color=palette[i], 
                    linestyle="--", 
                    lw=1)    
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * tailleCluster, f'{i}')
        ax.text( 0.05, 
                 y_lower + 0.5 * tailleCluster, 
                 f"{moyenne:0.2f}",
                 # rotation='vertical',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),)

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    # ax.set_title("The silhouette plot for the various clusters.")
    # ax.set_xlabel("The silhouette coefficient values")
    ax.set_xlabel("Silhouettes")
    ax.set_ylabel("")

    # The vertical line for average silhouette score of all the values
    moyenne = donnees.silhouettes.mean()
    ax.axvline(x=moyenne, color="red", linestyle="-", lw=2)
    ax.text( moyenne - 0.05, 
             donnees.shape[0]/3, 
             f"moyenne = {moyenne:0.2f}",
             rotation='vertical',
             color='red',   
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),);


def afficheSilhouettesClustersV(donnees, infoKMeans, tailleImage=12):
    clusters,_ = infoKMeans.shape
    fig,axes = plt.subplots(nrows=clusters,ncols=2,figsize=(tailleImage*2,tailleImage*clusters))
    for i in range(clusters):
        _, _ = afficheKMeansClusters(donnees, infoKMeans, cluster=2+i, ax=axes[i,0])
        afficheGraphiqueSilhouettes(retrouveDFCluster(infoKMeans, nrCluster = 2+i), ax=axes[i,1])

def afficheSilhouettesClustersH(donnees, infoKMeans, tailleImage=12):
    clusters,_ = infoKMeans.shape
    fig,axes = plt.subplots(nrows=2,ncols=clusters,figsize=(tailleImage*clusters,tailleImage*2))
    for i in range(clusters):
        afficheGraphiqueSilhouettes(retrouveDFCluster(infoKMeans, nrCluster = 2+i), ax=axes[0,i])
        _, _ = afficheKMeansClusters(donnees, infoKMeans, cluster=2+i, ax=axes[1,i])

def afficheSilhouettesClusters(donnees, infoKMeans, tailleImage=12):
    clusters,_ = infoKMeans.shape
    fig,axes = plt.subplots(nrows=1,ncols=clusters,figsize=(tailleImage*clusters,tailleImage))
    for i in range(clusters):
        afficheGraphiqueSilhouettes(retrouveDFCluster(infoKMeans, nrCluster = 2+i), ax=axes[i])

def bic_score(X: np.ndarray, labels: np.array):
    """
    BIC score for the goodness of fit of clusters.
    This Python function is translated from the Golang implementation by the author of the paper. 
    The original code is available here: https://github.com/bobhancock/goxmeans/blob/a78e909e374c6f97ddd04a239658c7c5b7365e5c/km.go#L778
    """

    n_points = len(labels)
    n_clusters = len(set(labels))
    n_dimensions = X.shape[1]

    n_parameters = (n_clusters - 1) + (n_dimensions * n_clusters) + 1

    loglikelihood = 0
    for label_name in set(labels):
        X_cluster = X[labels == label_name]
        n_points_cluster = len(X_cluster)
        centroid = np.mean(X_cluster, axis=0)
        variance = np.sum((X_cluster - centroid) ** 2) / (len(X_cluster) - 1)
        loglikelihood += \
          n_points_cluster * np.log(n_points_cluster) \
          - n_points_cluster * np.log(n_points) \
          - n_points_cluster * n_dimensions / 2 * np.log(2 * math.pi * variance) \
          - (n_points_cluster - 1) / 2

    bic = loglikelihood - (n_parameters / 2) * np.log(n_points)
        
    return bic


def executionKMeans(donnees, listeClusters=range(2,21)):
    infoExecutions = {'clusters':list(),
                      'itterations':list(),
                      'silhouette':list(),
                      'calinski_harabasz':list(),                        
                      'davies_bouldin':list(),                        
                      'bic':list(),                        
                      'wss':list(),
                      'classes':list(),
                      'centres':list(),
                      'silhouettes':list()
                     }


    for nbClusters in listeClusters :
        modelKMeans = KMeans(n_clusters=nbClusters, max_iter=1000,random_state=0)
        modelKMeans.fit(donnees)
        infoExecutions['clusters'].append(nbClusters)
        infoExecutions['wss'].append(modelKMeans.inertia_)
        infoExecutions['silhouettes'].append(silhouette_samples(donnees, modelKMeans.labels_))
        infoExecutions['classes'].append(modelKMeans.labels_)
        infoExecutions['itterations'].append(modelKMeans.n_iter_)
        infoExecutions['centres'].append(modelKMeans.cluster_centers_)
        infoExecutions['silhouette'].append(silhouette_score(donnees, modelKMeans.labels_))#plus c'est haut, mieux c'est
        infoExecutions['calinski_harabasz'].append(calinski_harabasz_score(donnees, modelKMeans.labels_))#plus c'est haut, mieux c'est
        infoExecutions['davies_bouldin'].append(davies_bouldin_score(donnees, modelKMeans.labels_))#plus c'est bas, mieux c'est
        infoExecutions['bic'].append(bic_score(donnees.values, modelKMeans.labels_))#plus c'est haut, mieux c'est
        
    return pd.DataFrame(infoExecutions)                      

def affichePredictionClusters(donnees, 
                              classes, 
                              taillePoints=200, 
                              tailleImage=12, 
                              ax=None, 
                              afficheLegend=None, 
                              afficheTicks=True, 
                              palette=palette,
                              outlier=outlier,
                              nomDonnees=None):
    colonnes = donnees.columns
    affichage = donnees.copy()
    affichage['Classe'] = classes
    nbClasses = len(affichage[affichage['Classe'] != 'outlier']['Classe'].unique())
                    
    
    if ax is None :
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(tailleImage,tailleImage))
    
    sns.scatterplot(x=colonnes[0],
                    y=colonnes[1], 
                    hue='Classe', 
                    data=affichage.sort_values('Classe'),
                    alpha=0.6,       
                    s=taillePoints,    
                    palette=palette[:nbClasses]+[outlier],
                    legend=afficheLegend,
                    ax=ax) ; 
    
    ax.text( affichage[colonnes[0]].max() * 0.85, 
             affichage[colonnes[1]].max() * 0.9, 
             f"{nbClasses:d}",
             color="#030aa7",   
             fontweight='bold', 
             fontsize='x-large',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),);
    
    if afficheTicks : 
        ax.set_xlabel('');ax.set_ylabel('');ax.set_xticks(());ax.set_yticks(())  
        
    if nomDonnees is not None :    
        ax.set_title(nomDonnees,fontweight='bold', fontsize='x-large')


def executionGaussianMixtureMetriques(donnees, listeClusters=range(2,21)):
    t1 = time.time()
    infoExecutions = {'clusters':list(),
                      'silhouette':list(),
                      'calinski_harabasz':list(),                        
                      'davies_bouldin':list(),                        
                      'bic':list(),                        
                      'classes':list(),
                      'silhouettes':list()
                     }


    for nbClusters in listeClusters :
        modelGaussianMixture = GaussianMixture(
                                    n_components=nbClusters,
                                    covariance_type='full',
                                    tol=1e-6,
                                    reg_covar=1e-8,
                                    max_iter=10000,
                                    n_init=50,
                                    init_params='k-means++',
                                    random_state=0,
                                    )
        
        classes = modelGaussianMixture.fit_predict(donnees)
        infoExecutions['clusters'].append(nbClusters)
        infoExecutions['silhouette'].append(silhouette_score(donnees, classes))#plus c'est haut, mieux c'est
        infoExecutions['calinski_harabasz'].append(calinski_harabasz_score(donnees, classes))#plus c'est haut, mieux c'est
        infoExecutions['davies_bouldin'].append(davies_bouldin_score(donnees, classes))#plus c'est bas, mieux c'est
        infoExecutions['bic'].append(modelGaussianMixture.bic(donnees))#plus c'est haut, mieux c'est
        infoExecutions['model'] = modelGaussianMixture
        infoExecutions['classes'].append(classes)
        infoExecutions['silhouettes'].append(silhouette_samples(donnees, classes))
        
    print(f'Execution dans :{(time.time()-t1):.2f}s')    
    return pd.DataFrame(infoExecutions)   