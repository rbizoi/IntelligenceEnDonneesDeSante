import pandas as pd, numpy as np, seaborn as sns, warnings, os, sys, time, copy as cp
from datetime import datetime as dt
from matplotlib import pyplot as plt

os.environ['SCIPY_ARRAY_API']='1'
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost  import XGBClassifier

from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedKFold, \
                                    RepeatedStratifiedKFold, LeavePOut, LeaveOneGroupOut, \
                                    LeavePGroupsOut, ShuffleSplit, StratifiedShuffleSplit, TimeSeriesSplit, GridSearchCV

from sklearn.preprocessing import StandardScaler,MinMaxScaler, label_binarize
from sklearn.feature_extraction import DictVectorizer

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier,LocalOutlierFactor

from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeCV

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, RationalQuadratic, ExpSineSquared, DotProduct, Matern, WhiteKernel

from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB

from sklearn.metrics import make_scorer, confusion_matrix, roc_curve, auc, accuracy_score, log_loss, hamming_loss, \
                            precision_score, recall_score, f1_score, fbeta_score, jaccard_score, \
                            precision_recall_curve, average_precision_score, balanced_accuracy_score, \
                            classification_report, roc_auc_score, zero_one_loss, multilabel_confusion_matrix, matthews_corrcoef


from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble   import IsolationForest

from sklearn.svm        import OneClassSVM, NuSVC, SVC

from yellowbrick.model_selection import RFECV

sys.path.append(os.path.abspath('../outils/'))
from prjFormation import palette, outlier

def initDictionnaireClassificateursC(arbres = 128):
    noms = [
            'Random_Forest',
            'AdaBoost',
            'LogisticRegression',
            'Stochastic_GD',
            'Gaussian_Process',
            'Nearest_Neighbors',
            'Linear_SVM', 
            'Linear_NuSVM', 
            'Radial_NuSVM',
            'Poly_NuSVM',
            'GaussianNaiveBayes',
            'QuadraticDiscriminant',
            'Neural_Net',
            'LightGBM',
            'XGBoost'
        ]

    classificateurs = [
                RandomForestClassifier(
                    max_depth=6,
                    max_features=3,
                    min_samples_split=4,
                    n_estimators=arbres,
                    n_jobs=-1
                ),
                AdaBoostClassifier(
                    n_estimators=arbres
                ),    
                LogisticRegression(
                    C=0.81113,
                    max_iter=3000,
                    penalty='l2',
                    solver='lbfgs',
                    n_jobs=-1
                ),
                SGDClassifier(
                    loss='log_loss', 
                    alpha=0.01, 
                    max_iter=200, 
                    fit_intercept=True
                ),
                GaussianProcessClassifier(
                    n_jobs=-1
                ),
                KNeighborsClassifier(
                    algorithm='ball_tree',
                    n_neighbors=17,
                    p=1,
                    weights='distance',
                    n_jobs=-1
                ),
                SVC(
                    kernel="linear" , 
                    C=10, #0.025, 
                    probability=True),
                NuSVC(
                    kernel="linear",
                    nu=0.1,
                    probability=True,
                ),
                NuSVC(
                    kernel="rbf",
                    gamma=0.5,
                    nu=0.195,
                    probability=True,
                ),
                NuSVC(
                    kernel="poly",
                    degree=3,
                    nu=0.1,
                    probability=True,
                ),
                GaussianNB(),
                QuadraticDiscriminantAnalysis(),
                MLPClassifier(
                    alpha=1
                ),
                LGBMClassifier(
                    learning_rate=0.1,
                    n_estimators=arbres,
                    num_leaves=20,
                    reg_alpha=0.1,
                    reg_lambda=20,
                    min_child_samples = 10,
                    min_split_gain = 0.01
                ),
                XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='auc',
                    n_estimators=arbres,
                    max_depth=6,
                    use_label_encoder=False
                )                
            ]

    return {nom:{'classicateur':OneVsRestClassifier(classicateur), 'couleur':couleur} 
            for nom, classicateur, couleur in zip(noms, classificateurs, palette)}   


def affichageROC(nom, classificateur, X_test, y_test):
    plt.figure(figsize=(36,36))

    score = classificateur.score(X_test, y_test)*100
    print(f'{nom:17s}'+(' %.4f' % score).lstrip('0'),end='\t--\t')
    probas = classificateur.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test.ravel(), probas[:, 1])
    roc_auc = auc(fpr, tpr)
    print ("Area under the ROC curve : %0.4f" % roc_auc,end='\t--\t')
    plt.scatter(fpr, tpr, color='blue')
    plt.plot(fpr, tpr, label=nom + '(AUC = %0.4f)' % roc_auc)
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    for i,t in enumerate(thresholds*100):
        if i > 1 :
            
            # plt.text(fpr[i], tpr[i]+0.01,str(t.round(2))) 
            plt.text(fpr[i], tpr[i]+0.001,f'{t:0.2f}({int(fpr[i]*100):02d})', rotation= 30, alpha=0.6) 

    plt.xlabel('Le taux de faux Positifs-(1 - Spécificité) = VN / (FP + VN)',size=18)
    plt.ylabel('Le taux de vrais positifs-Sensibilité = VP / (VP + FN)',size=18)
    plt.title('La courbe ROC (Receiver Operating Caracteristic)',size=20)
    plt.legend(loc="lower right")       

def calculMetriques(clasificateur, X_test, y_test):
    metriques, resultats = dict(),dict()

    resultats['observations'] = y_test
    resultats['predictions']  = clasificateur.predict(X_test)
    resultats['probabilites'] = clasificateur.predict_proba(X_test)[:,1]
    
    resultats['fauxPositifsR'], resultats['vraisPositifsR'], resultats['probabilitesR'] = \
                   roc_curve(resultats['observations'], resultats['probabilites'])

    resultats['precisionsPR'], resultats['rappelsPR'], resultats['tauxPR'] = \
                                 precision_recall_curve(resultats['observations'], resultats['probabilites'])
    
    metriques['avgPrecRec'] = average_precision_score(resultats['observations'], resultats['probabilites'])    
    
    metriques['aucROC']          = auc(resultats['fauxPositifsR'], resultats['vraisPositifsR'])
    metriques['accuracy']        = accuracy_score(resultats['observations'],resultats['predictions'])
    metriques['logloss']         = log_loss(resultats['observations'],resultats['predictions'])
    metriques['hammingloss']     = hamming_loss(resultats['observations'],resultats['predictions'])
    metriques['precision']       = precision_score(resultats['observations'],resultats['predictions'])
    metriques['sensibilite']     = recall_score(resultats['observations'],resultats['predictions'])
    metriques['f1']              = f1_score(resultats['observations'],resultats['predictions'])
    metriques['fbeta05']         = fbeta_score(resultats['observations'],resultats['predictions'], beta=0.5)
    metriques['fbeta2']          = fbeta_score(resultats['observations'],resultats['predictions'], beta=2)
    metriques['jaccard']         = jaccard_score(resultats['observations'],resultats['predictions'])
    metriques['matthews']        = matthews_corrcoef(resultats['observations'],resultats['predictions'])    
    
    metriques['vrais_negatifs']  = confusion_matrix(resultats['observations'], resultats['predictions'])[0, 0]
    metriques['faux_positifs']   = confusion_matrix(resultats['observations'], resultats['predictions'])[0, 1]
    metriques['faux_negatifs']   = confusion_matrix(resultats['observations'], resultats['predictions'])[1, 0]
    metriques['vrais_positifs']  = confusion_matrix(resultats['observations'], resultats['predictions'])[1, 1] 
    
    metriques['specificite']     = metriques['vrais_negatifs']/(metriques['faux_positifs']+metriques['vrais_negatifs'])
    
    return metriques, resultats

def affichageCourbesROCetPR(metriques,resultats,classificateursDict):
    lw = 1
    fig,(ax0,ax1) = plt.subplots(ncols=2,figsize=(36,18))
    for nom in classificateursDict:        
        graph0 = sns.lineplot(x        = resultats[nom]['fauxPositifsR'], 
                             y         = resultats[nom]['vraisPositifsR'], 
                             estimator = None, 
                             lw        = 1, 
                             color     = classificateursDict[nom]['couleur'],
                             ax        = ax0,
                             label=f"{nom} (AUC = {metriques[nom]['aucROC']:0.6f})")
        graph0.fill_between(resultats[nom]['fauxPositifsR'], 
                            resultats[nom]['vraisPositifsR'], 
                            step='post', alpha=0.05)
        
        graph1 = sns.lineplot(x        = resultats[nom]['rappelsPR'], 
                             y         = resultats[nom]['precisionsPR'], 
                             estimator = None, 
                             lw        = 1, 
                             color     = classificateursDict[nom]['couleur'],
                             ax        = ax1,
                             label=f"{nom} (AP = {metriques[nom]['avgPrecRec']:0.6f})")       
        
        graph1.fill_between(resultats[nom]['rappelsPR'], 
                            resultats[nom]['precisionsPR'], 
                            step='post', 
                            alpha=0.05)
        
    graph0.plot([0, 1], [0, 1], 'k--')
    sns.move_legend(ax0, "lower right", bbox_to_anchor=(.85, .05), frameon=False)
    graph0.set_xlabel('Faux Positifs - (1 - Spécificité) = VN / (FP + VN)')
    graph0.set_ylabel('Vrais Positifs - Sensibilité = VP / (VP + FN)')  

    f_scores = np.linspace(0.2, 0.9, num=7)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = ax1.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        ax1.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))    
    
    sns.move_legend(ax1, "lower right", bbox_to_anchor=(.85, .05), frameon=False)
    ax1.plot([0, 1], [0, 1], 'k--')
    
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    
    graph1.set_xlabel('Sensibilité(Rappel) = VP / (VP + FN)')
    graph1.set_ylabel('Précision = VP / (VP + FP)');     

def essaiApprentissageComparaisonClassificateurs( 
                       classificateursDict, 
                       X_train, 
                       y_train,
                       X_test, 
                       y_test, 
                       nom_essai,
                       apprentissage=True, 
                       modelRFE=None):
    t0 = time.time()  

    metriques,resultats = { nom:{} for nom in classificateursDict },\
                          { nom:{} for nom in classificateursDict }
    
    for nom in classificateursDict:        
        t1 = time.time()  
        if modelRFE is None :
            if apprentissage : classificateursDict[nom]['classicateur'].fit(X_train, y_train)
        else:
            classificateursDict[nom]['classicateur'].fit(X_train[X_train.columns[modelRFE[nom].support_]], 
                                                         y_train)
            
        if modelRFE is None :
            metriques[nom],resultats[nom] = calculMetriques(classificateursDict[nom]['classicateur'], X_test, y_test)
        else:
            metriques[nom],resultats[nom] = calculMetriques(classificateursDict[nom]['classicateur'], 
                                                        X_test[X_test.columns[modelRFE[nom].support_]],
                                                        y_test)
        
        metriques[nom]['essai'] = resultats[nom]['essai'] = nom_essai

        print(f"{nom:21s} {metriques[nom]['accuracy']:0.4f} -- Area under the ROC curve : {metriques[nom]['aucROC']:0.4f} -- Exécution : {(time.time() - t1):0.2f}")

    print(f"Exécution : {(time.time() - t1):0.2f}")
    
    metriquesDF = pd.DataFrame(metriques).T
    metriquesDF = metriquesDF.reset_index().rename(columns={'index':'classicateur'}).set_index('classicateur')
    metriquesDF.sort_values('aucROC',ascending=False, inplace=True)    
    resultatsDF = pd.DataFrame(resultats).T
    resultatsDF = resultatsDF.reset_index().rename(columns={'index':'classicateur'}).set_index('classicateur')
    
    return metriquesDF,resultatsDF,metriques,resultats 