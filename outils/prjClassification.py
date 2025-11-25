import pandas as pd, numpy as np, math, seaborn as sns, warnings, os, sys, time, copy as cp, pickle

from datetime import datetime as dt
from matplotlib import pyplot as plt

import matplotlib.font_manager as fm
import plotly.express as px

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import NuSVC, SVC, OneClassSVM
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, RationalQuadratic, ExpSineSquared, DotProduct, Matern, WhiteKernel

from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from lightgbm import LGBMClassifier
from xgboost  import XGBClassifier
from sklearn.metrics import roc_curve, auc

from sklearn.metrics  import make_scorer, confusion_matrix, roc_curve, auc, accuracy_score, log_loss, hamming_loss, \
                             precision_score, recall_score, f1_score, fbeta_score, jaccard_score,  \
                             precision_recall_curve, average_precision_score, precision_recall_fscore_support, matthews_corrcoef

sys.path.append(os.path.abspath('../outils/'))
from prjFormation import palette, outlier

def affichagePartageDonnees(donnees, palette, ax, afficheTitre=True):
    markers = {"apprentissage": "o", "test": "s"}
    sns.scatterplot( x='X1',
                     y='X2', 
                     hue='cible', 
                     data=donnees,
                     alpha=0.6,       
                     s=160,    
                     palette=palette,
                     style='échantillon',
                     markers=markers,
                     legend=None,
                     ax=ax);
    ax.set_xlabel('');ax.set_ylabel('');
    ax.set_xticks(());ax.set_yticks(())  
    if afficheTitre:
        ax.set_title(f"apprentissage{donnees[donnees['échantillon']=='apprentissage'].shape[0]}\n"+
                     f"test{donnees[donnees['échantillon']=='test'].shape[0]}",
                     fontweight='bold', 
                     fontsize='x-large');

def calculMatriceConfusionMetriques(donnees, seuil, Prevalence):
    donnees.Prediction = donnees.Probabilite.apply(lambda x: 0 if x <= seuil else 1)
    
    vraisNegatifs     = confusion_matrix(donnees.Observation, donnees.Prediction)[0, 0]
    fauxPositifs      = confusion_matrix(donnees.Observation, donnees.Prediction)[0, 1]
    fauxNegatifs      = confusion_matrix(donnees.Observation, donnees.Prediction)[1, 0]
    vraisPositifs     = confusion_matrix(donnees.Observation, donnees.Prediction)[1, 1]

    Sensibilite       = vraisPositifs / (vraisPositifs + fauxNegatifs)
    Specificite       = vraisNegatifs / (fauxPositifs  + vraisNegatifs)
    Precision         = vraisPositifs / (vraisPositifs + fauxPositifs) # 
    F1Score           = 2 * Precision * Sensibilite / (Precision + Sensibilite) # 
    
    return [Sensibilite,Specificite,Precision,F1Score,
            vraisPositifs,vraisNegatifs,fauxNegatifs,fauxPositifs]


def calculMetriquesCourbes(donnees, seuils, Prevalence):
    Sensibilite,Specificite,Precision,F1Score,vraisPositifs,vraisNegatifs,fauxNegatifs,fauxPositifs = \
                            dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict()
    
    for t in seuils:
        Sensibilite[t],Specificite[t],Precision[t],F1Score[t],\
        vraisPositifs[t],vraisNegatifs[t],\
        fauxNegatifs[t],fauxPositifs[t] = calculMatriceConfusionMetriques(donnees, t, Prevalence)   
    
    metriques = pd.concat([pd.Series(Sensibilite,name='Sensibilite'),
                            pd.Series(Specificite,name='Specificite'),
                            pd.Series(Precision,name='Precision'),
                            pd.Series(F1Score,name='F1Score'),
                            pd.Series(vraisPositifs,name='vraisPositifs'),
                            pd.Series(vraisNegatifs,name='vraisNegatifs'),
                            pd.Series(fauxNegatifs,name='fauxNegatifs'),
                            pd.Series(fauxPositifs,name='fauxPositifs')],axis=1)
    metriques.reset_index(inplace=True)
    metriques.columns = ['Probabilite'] + list(metriques.columns[1:])
    metriques.loc[0,['Precision']],metriques.loc[0,['F1Score']] = 1,0
    return metriques 


def affichageCourbeROC(fauxPositifs,vraisPositifs, roc_auc, palette, ax, nom, affichePoints = True, afficheTitre = False):
    graph = sns.lineplot(x         = fauxPositifs, 
                         y         = vraisPositifs, 
                         estimator = None, 
                         lw        = 2, 
                         color     = palette[1],
                         ax        = ax,
                         label=f'Courbe ROC (AUC = {roc_auc:0.2f})')

    if affichePoints :
        sns.scatterplot     (x         = fauxPositifs, 
                             y         = vraisPositifs, 
                             alpha     = 0.8,   
                             s         = 50,
                             color     = palette[0],
                             ax        = graph,);
    
    graph.fill_between(fauxPositifs, vraisPositifs, step='post', alpha=0.1)
    graph.plot([0, 1], [0, 1], 'k--')
    sns.move_legend(ax, "lower right", bbox_to_anchor=(.85, .05), frameon=False)
    
    # graph.set_xlabel('Taux de faux Positifs-(1 - Spécificité) = VN / (FP + VN)')
    # graph.set_ylabel('Taux de vrais positifs-Sensibilité = VP / (VP + FN)')  
    graph.set_xlabel('Faux Positifs')
    graph.set_ylabel('Vrais Positifs')  
    # if afficheTitre : graph.set_title(f"La courbe ROC (Receiver Operating Caracteristic)\n{nom}",size=32);
    if afficheTitre : graph.set_title(f"La courbe ROC {nom}",size=32);    


def affichageCourbePR(precisions, sensibilites, pr_auc, palette, ax, nom, affichePoints = True, afficheTitre = False):
    graph = sns.lineplot(x         = sensibilites, 
                         y         = precisions, 
                         estimator = None, 
                         lw        = 2, 
                         color     = palette[1],
                         ax        = ax,
                         label=f'Courbe PR (AUC = {pr_auc:0.2f})')
    
    if affichePoints :
        sns.scatterplot     (x         = sensibilites, 
                             y         = precisions, 
                             alpha     = 0.8,   
                             s         = 50,
                             color     = palette[0],
                             ax        = graph,);

    f_scores = np.linspace(0.2, 0.9, num=7)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        ax.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))    
    
    graph.fill_between(sensibilites, precisions, step='post', alpha=0.1)
    sns.move_legend(ax, "lower right", bbox_to_anchor=(.85, .05), frameon=False)
    ax.plot([0, 1], [0, 1], 'k--')
    
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    # sns.move_legend(ax, "lower right", bbox_to_anchor=(.85, .05), frameon=False)
    
    # graph.set_xlabel('Sensibilité(Rappel) = VP / (VP + FN)')
    # graph.set_ylabel('Précision = VP / (VP + FP)');
    graph.set_xlabel('Sensibilité(Rappel)')
    graph.set_ylabel('Précision');    
    if afficheTitre : graph.set_title(f"Courbe Précision-Rappel {nom}",size=32)
   
def affichageSurfaceDecision(X_train, X_test, y_train, y_test, classificateur, ax, titre=None):
    h = .03  # step size in the mesh
    X = pd.concat([X_train,X_test]).values
    
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    score = classificateur.score(X_test, y_test)
    Z = classificateur.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    
    ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap=cm_bright,s=160, edgecolors='k', alpha=0.8, )
    ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap=cm_bright,s=160, edgecolors='k', alpha=0.3, marker='s');
    
    ax.text(xx.max() - .3, yy.min() + .3, f'accuracy {score*100:.2f}%', size=36, 
            bbox=dict(boxstyle="round", alpha=0.8, facecolor="white"),
            horizontalalignment='right')
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_ylabel('');ax.set_ylabel('');ax.set_xticks(());ax.set_yticks(())  
    if titre is not None: ax.set_title(titre,fontweight='bold', fontsize='x-large');


def executionAffichageExecutionMultiple( classificateur,
                                         nomClassificateur,
                                         donnees,
                                         X_train, 
                                         X_test, 
                                         y_train, 
                                         y_test, 
                                         affiche_titre = True, 
                                         sauvegarde=True, 
                                         affichePoints = True, 
                                         palette=None, 
                                         projet=None):
    fig,(ax0, ax1, ax2, ax3) = plt.subplots(nrows=1,ncols=4,figsize=(72,16))
    affichagePartageDonnees(donnees, palette, ax0, afficheTitre=False)
    classificateur.fit(X_train, y_train)
    probabilites = classificateur.predict_proba(X_test)
    predictions  = classificateur.predict(X_test) # 50%
    observations = y_test.ravel()
    fauxPositifs, vraisPositifs, seuils = roc_curve(observations,probabilites[:, 1])
    roc_auc = auc(fauxPositifs, vraisPositifs)

    precisions, sensibilites, seuilsPR = precision_recall_curve(observations,probabilites[:, 1])
    pr_auc = average_precision_score(observations,probabilites[:, 1])

    affichageSurfaceDecision(X_train, X_test, y_train, y_test, classificateur, ax1)
    
    affichageCourbeROC(fauxPositifs,vraisPositifs, roc_auc, palette, ax2, nomClassificateur, affichePoints = affichePoints)

    affichageCourbePR(precisions, sensibilites, pr_auc, palette, ax3, nomClassificateur, affichePoints = affichePoints)

    if affiche_titre : fig.suptitle(f"Expérience - {nomClassificateur}", fontsize=64)
    if sauvegarde : projet.sauvegarderImage( f"Expérience-{nomClassificateur}")   
    
    donneesInference = pd.DataFrame({'Probabilite':probabilites[:,1],
                      'Observation':observations,
                      'Prediction':predictions
                    }).sort_values(by='Probabilite',ascending=False)
    Prevalence = donneesInference.Observation.sum()/donneesInference.Observation.count()
    return donneesInference, calculMetriquesCourbes(donneesInference, seuils[1:], Prevalence)



