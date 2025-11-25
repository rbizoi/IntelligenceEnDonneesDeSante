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

    return {nom:{'classicateur':classicateur, 'couleur':couleur} 
            for nom, classicateur, couleur in zip(noms, classificateurs, palette)}   

def initDictionnaireClassificateurs(arbres = 128):
    names = [
            'Random_Forest',
            'AdaBoost',
            'LightGBM',
            'XGBoost',
            'LogisticRegression',
            'Stochastic_GD',
            'Gaussian_Process',
            'Nearest_Neighbors',
            'Linear_SVM', 
            'Radial_NuSVM',
            'Poly_NuSVM',
            'GaussianNaiveBayes',
            'QuadraticDiscriminant',
            'Neural_Net'
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
            ]

    #classificateursDict = {name:clf for name, clf in zip(names, classifiers)}   
    return {name:clf for name, clf in zip(names, classificateurs)}   


def executionEssaiComparaisonClassificateurs( 
                       classificateursDict, 
                       X_train, 
                       y_train,
                       X_test, 
                       y_test, 
                       couleurs,
                       nom_essai,
                       projet):
    np.random.seed(123456)
    t0 = time.time()  
    # h = .02  # step size in the mesh
    lw = 1

    plt.figure(figsize=(36,18))

    r_acc,r_aucROC = dict(),dict()
    fauxPositifs, vraisPositifs, probabilites = dict(),dict(),dict()
    accuracy,logloss,hammingloss,precision,sensibilite,specificite,f1,jaccard = dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict()
    prec, rec, tauxPR, avgPrecRec =  dict(),dict(),dict(),dict()
    vrais_negatifs, faux_positifs, faux_negatifs, vrais_positifs = dict(),dict(),dict(),dict()

    for i, nom in enumerate(sorted(classificateursDict.keys())):    
        t1 = time.time()  
        recapitulatif = f'{nom:21s} '
        #print(f'{nom:21s}',end=' ')
        classificateursDict[nom].fit(X_train, y_train)

        y_probas = classificateursDict[nom].predict_proba(X_test)
        y_pred = classificateursDict[nom].predict(X_test)
        fauxPositifs[nom], vraisPositifs[nom], probabilites[nom] = roc_curve(y_test.ravel(), y_probas[:, 1])

        r_aucROC[nom]     = auc(fauxPositifs[nom], vraisPositifs[nom])
        accuracy[nom]     = accuracy_score(y_test,y_pred)
        logloss[nom]      = log_loss(y_test,y_pred)
        hammingloss[nom]  = hamming_loss(y_test,y_pred)
        precision[nom]    = precision_score(y_test,y_pred)
        sensibilite[nom]  = recall_score(y_test,y_pred)
        f1[nom]           = f1_score(y_test,y_pred)
        jaccard[nom]      = jaccard_score(y_test,y_pred)


        vrais_negatifs[nom]        = confusion_matrix(y_test, y_pred)[0, 0]
        faux_positifs[nom]         = confusion_matrix(y_test, y_pred)[0, 1]
        faux_negatifs[nom]         = confusion_matrix(y_test, y_pred)[1, 0]
        vrais_positifs[nom]        = confusion_matrix(y_test, y_pred)[1, 1]        

        specificite[nom]           = vrais_negatifs[nom]/(faux_positifs[nom]+vrais_negatifs[nom]) 

        prec[nom], rec[nom], tauxPR[nom] = precision_recall_curve(y_test.ravel(), y_probas[:, 1])
        avgPrecRec[nom] = average_precision_score(y_test.ravel(), y_probas[:, 1])

        # print(f'{nom:21s}'+(' %.4f' % accuracy[nom]).lstrip('0'),end='\t--\t')
        recapitulatif += (' %.4f' % accuracy[nom]).lstrip('0') + '\t--\t'
        recapitulatif += "Area under the ROC curve : %0.4f" % r_aucROC[nom] + '\t--\t'
        recapitulatif += 'Exécution  :'+('%.2fs' % (time.time() - t1)).lstrip('0')
        
        #print((' %.4f' % accuracy[nom]).lstrip('0'),end='\t--\t')
        #print ("Area under the ROC curve : %0.4f" % r_aucROC[nom],end='\t--\t')
        #print('Exécution  :'+('%.2fs' % (time.time() - t1)).lstrip('0'))
        print(recapitulatif)

        plt.subplot(1, 2, 1)
        plt.plot(fauxPositifs[nom], vraisPositifs[nom], color=couleurs[i], label=nom + '(AUC = %0.4f)' % r_aucROC[nom])

        plt.subplot(1, 2, 2)
        plt.step(rec[nom], prec[nom], where='post', color=couleurs[i], label=f"{nom}(AP = {avgPrecRec[nom]:0.8f})")#alpha=0.8, 
        plt.fill_between(rec[nom], prec[nom], step='post', alpha=0.05)



    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Le taux de faux Positifs-(1 - Spécificité) = VN / (FP + VN)',size=18)
    plt.ylabel('Le taux de vrais positifs-Sensibilité = VP / (VP + FN)',size=18)
    plt.title('La courbe ROC (Receiver Operating Caracteristic)',size=20)
    plt.legend(loc="lower right")    

    plt.subplot(1, 2, 2)
    f_scores = np.linspace(0.2, 0.9, num=7)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Sensibilité(Rappel) = VP / (VP + FN)', size=18)
    plt.ylabel('Précision = VP / (VP + FP)', size=18)      
    plt.title('La courbe Précision-Rappel',size=20)
    plt.legend(loc="lower left")    

    projet.sauvegarderImage(f"Les courbes ROC et Précision-Rappel--{nom_essai}")    
    plt.show()

    print('Exécution  :'+('%.2fs' % (time.time() - t0)).lstrip('0'))

    resultat = pd.DataFrame(pd.Series(r_aucROC), columns=["aucROC"])
    resultat['avgPrecRec']      = pd.Series(avgPrecRec   )
    resultat['accuracy']        = pd.Series(accuracy   )
    resultat['f1']              = pd.Series(f1         )
    resultat['precision']       = pd.Series(precision  )
    resultat['sensibilite']     = pd.Series(sensibilite)
    resultat['specificite']     = pd.Series(specificite)
    resultat['logloss']         = pd.Series(logloss    )
    resultat['hammingloss']     = pd.Series(hammingloss)
    resultat['jaccard']         = pd.Series(jaccard    )
    resultat["vrais_positifs"]  = pd.Series(vrais_positifs)
    resultat["vrais_negatifs"]  = pd.Series(vrais_negatifs)
    resultat["faux_positifs"]   = pd.Series(faux_positifs)
    resultat["faux_negatifs"]   = pd.Series(faux_negatifs)

    resultat['essai']           = nom_essai
    resultat                    = resultat.reset_index().rename(columns={'index':'Classifieur'}).set_index('Classifieur')
    
    resultat.sort_values('f1',ascending=False, inplace=True)
    return resultat

def affichageEvolutionMetriques(resultats, metrique, palette):
    fig, ax = plt.subplots(figsize=(24,12))
    affichage = resultats.reset_index()
    for i, nom in enumerate(affichage['Classifieur'].sort_values().unique()):
        graph = sns.lineplot( x         = 'essai', 
                              y         = metrique, 
                              data      = affichage[affichage.Classifieur == nom], 
                              estimator = None, 
                              lw        = 2, 
                              # ci        = None,
                              label     = nom,
                              color     = palette[i],
                              ax        = ax)
        sns.scatterplot( x     = 'essai', 
                         y     = metrique, 
                         data  = affichage[affichage.Classifieur == nom], 
                         alpha = 0.8,   
                         s     = 100,
                         # ci    = None, 
                         color = palette[i],
                         ax    = ax,);
    ax.set_title(f"Evolution du {metrique} suivant les traitements",fontsize = 36);          

def affichageEvolutionMetriquesC(metriques, metrique, classificateursDict, rotation=0):

    palette = {nom:classificateursDict[nom]['couleur'] for nom in classificateursDict}
    fig, ax = plt.subplots(figsize=(24,12))
    affichage = metriques.reset_index()
    for nom in affichage['classicateur'].sort_values().unique():
        graph = sns.lineplot( x         = 'essai', 
                              y         = metrique, 
                              data      = affichage[affichage.classicateur == nom], 
                              estimator = None, 
                              lw        = 2, 
                              # ci        = None,
                              label     = nom,
                              color     = palette[nom],
                              ax        = ax)
        sns.scatterplot( x     = 'essai', 
                         y     = metrique, 
                         data  = affichage[affichage.classicateur == nom], 
                         alpha = 0.8,   
                         s     = 100,
                         # ci    = None, 
                         color = palette[nom],
                         ax    = ax,);
    ax.set_title(f"Evolution du {metrique} suivant les traitements",fontsize = 36);  
    ax.set_xticklabels(affichage.essai.unique(),rotation=rotation);


def affichageEvolutionMetriquesTestVal(metriques, metrique, classificateursDict, rotation=0):
    palette = {nom:classificateursDict[nom]['couleur'] for nom in classificateursDict}
    affichage = metriques.copy().reset_index()
    classifieurs = affichage['classicateur'].sort_values().unique()
    if 'échantillon' not in affichage.columns : return
        
    fig, ax = plt.subplots(figsize=(24,12))
    for nom in affichage['classicateur'].sort_values().unique():
        graph = sns.lineplot( x         = 'essai', 
                              y         = metrique, 
                              data      = affichage[(affichage['classicateur'] == nom)&(affichage['échantillon'] == 'test')], 
                              estimator = None, 
                              lw        = 2, 
                              # ci        = None,
                              label     = nom +'-'+ 'test',
                              color     = palette[nom.split('-')[0]],
                              ax        = ax)

        graph = sns.lineplot( x         = 'essai', 
                              y         = metrique, 
                              data      = affichage[(affichage['classicateur'] == nom)&(affichage['échantillon'] == 'validation')], 
                              estimator = None, 
                              lw        = 1, 
                              # ci        = None,
                              label     = nom +'-'+ 'validation',
                              color     = palette[nom.split('-')[0]],
                              linestyle = '--',
                              ax        = ax)

        
        sns.scatterplot( x     = 'essai', 
                         y     = metrique, 
                         data  = affichage[affichage['classicateur'] == nom], 
                         alpha = 0.8,   
                         s     = 100,
                         # ci    = None, 
                         color = palette[nom.split('-')[0]],
                         ax    = ax,);
    ax.set_title(f"Evolution du {metrique} suivant les traitements",fontsize = 36);  
    ax.set_xticklabels(affichage.essai.unique(),rotation=rotation);

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

def executeValidationCroisee(X, y, classificateur, validation):
    n_samples, n_features = X.shape
    cv = validation
    
    classifieurs = []
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(18,18))
    i = 0
    for train, test in cv.split(X, y):
        classifier = cp.deepcopy(classificateur)
        classifier.fit(X[train], y[train])
        classifieurs.append(classifier)
        
        probas_ = classifier.predict_proba(X[test])
        
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.8f)' % (i, roc_auc))
    
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='AUC=0.5', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Courbe ROC moyenne(AUC = %0.8f $\pm$ %0.8f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 écart type')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Taux de faux Positifs-(1 - Spécificité) = VN / (FP + VN)',size=18)
    plt.ylabel('Taux de vrais positifs-Sensibilité = VP / (VP + FN)',size=18)
    plt.title('Courbe ROC (Receiver Operating Caracteristic)',size=20)
    plt.legend(loc="lower right")   
    plt.show()
    return classifieurs    


def afficheSeparationsBinaires(donnees, palette, ax, cible, valeurs=['','']) :
    affichage = donnees.reset_index().copy()
    affichage.sort_values([cible,'échantillon'],inplace=True)
    valeurs=affichage['échantillon'].sort_values().unique()

    graph = sns.countplot(x=cible,
                          hue='échantillon',
                          data=affichage, 
                          palette=palette,
                          ax=ax
                         );
    l = len(affichage['échantillon'].unique())
    affichage = affichage.groupby('échantillon').agg({cible:'count'})
    for patche in graph.patches:
        if patche.get_height() > 0 :
            graph.text(
                        patche.get_x() + patche.get_width() / 2 ,
                        affichage[cible].mean()/10, 
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
    graph.legend( bbox_to_anchor=(.55, .95), 
                  loc='upper left', 
                  borderaxespad=0, 
                  labels=[f"{valeur} ({affichage.loc[valeur,cible]})" for valeur in valeurs]);

def calculRFE(classificateursDict, X, y):
    modelRFE = {}
    for nom in classificateursDict:
        modelRFE[nom] = RFE(classificateursDict[nom])
        modelRFE[nom].fit(X, y)
    return modelRFE
    
def apprentissageRFECV(model, X, y):
    t0 = time.time()  
    min_features_to_select = 1  # Minimum number of features to consider
    modelRFE = RFECV(
        estimator=model,
        step=1,
        cv=StratifiedKFold(5),
        scoring='roc_auc',
        min_features_to_select=min_features_to_select,
        n_jobs=2,
    )
    modelRFE.fit(X, y)
    return modelRFE

def calculRFECV(classificateursDict, X, y):
    algorithmesRFECV = {}
    for nom in classificateursDict:  
        t0 = time.time()  
        algorithmesRFECV[nom] = apprentissageRFECV(classificateursDict[nom]['classicateur'], X, y)
        algorithmesRFECV[nom].fit(X, y)
        print(f"{nom} : Optimal number of features: {algorithmesRFECV[nom].n_features_} Exécution : {(time.time() - t0):0.2f}")        
    return algorithmesRFECV     
    
def executionEssaiComparaisonClassificateursRFE( 
                       classificateursDict,
                       modelRFE,
                       X_train, 
                       y_train,
                       X_test, 
                       y_test, 
                       couleurs,
                       nom_essai,
                       projet):
    np.random.seed(123456)
    t0 = time.time()  
    # h = .02  # step size in the mesh
    lw = 1

    plt.figure(figsize=(36,18))

    r_acc,r_aucROC = dict(),dict()
    fauxPositifs, vraisPositifs, probabilites = dict(),dict(),dict()
    accuracy,logloss,hammingloss,precision,sensibilite,specificite,f1,jaccard = dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict()
    prec, rec, tauxPR, avgPrecRec =  dict(),dict(),dict(),dict()
    vrais_negatifs, faux_positifs, faux_negatifs, vrais_positifs = dict(),dict(),dict(),dict()

    for i, nom in enumerate(sorted(classificateursDict.keys())):    
        t1 = time.time()  
        recapitulatif = f'{nom:21s} '
        #print(f'{nom:21s}',end=' ')
        #classificateursDict[nom].fit(X_train, y_train)
        classificateursDict[nom].fit(X_train[X_train.columns[modelRFE[nom].support_]], 
                                     y_train)

        y_probas = classificateursDict[nom].predict_proba(X_test[X_train.columns[modelRFE[nom].support_]])
        y_pred = classificateursDict[nom].predict(X_test[X_train.columns[modelRFE[nom].support_]])
        fauxPositifs[nom], vraisPositifs[nom], probabilites[nom] = roc_curve(y_test.ravel(), y_probas[:, 1])

        r_aucROC[nom]     = auc(fauxPositifs[nom], vraisPositifs[nom])
        accuracy[nom]     = accuracy_score(y_test,y_pred)
        logloss[nom]      = log_loss(y_test,y_pred)
        hammingloss[nom]  = hamming_loss(y_test,y_pred)
        precision[nom]    = precision_score(y_test,y_pred)
        sensibilite[nom]  = recall_score(y_test,y_pred)
        f1[nom]           = f1_score(y_test,y_pred)
        jaccard[nom]      = jaccard_score(y_test,y_pred)


        vrais_negatifs[nom]        = confusion_matrix(y_test, y_pred)[0, 0]
        faux_positifs[nom]         = confusion_matrix(y_test, y_pred)[0, 1]
        faux_negatifs[nom]         = confusion_matrix(y_test, y_pred)[1, 0]
        vrais_positifs[nom]        = confusion_matrix(y_test, y_pred)[1, 1]        

        specificite[nom]           = vrais_negatifs[nom]/(faux_positifs[nom]+vrais_negatifs[nom]) 

        prec[nom], rec[nom], tauxPR[nom] = precision_recall_curve(y_test.ravel(), y_probas[:, 1])
        avgPrecRec[nom] = average_precision_score(y_test.ravel(), y_probas[:, 1])

        # print(f'{nom:21s}'+(' %.4f' % accuracy[nom]).lstrip('0'),end='\t--\t')
        recapitulatif += (' %.4f' % accuracy[nom]).lstrip('0') + '\t--\t'
        recapitulatif += "Area under the ROC curve : %0.4f" % r_aucROC[nom] + '\t--\t'
        recapitulatif += 'Exécution  :'+('%.2fs' % (time.time() - t1)).lstrip('0')
        
        #print((' %.4f' % accuracy[nom]).lstrip('0'),end='\t--\t')
        #print ("Area under the ROC curve : %0.4f" % r_aucROC[nom],end='\t--\t')
        #print('Exécution  :'+('%.2fs' % (time.time() - t1)).lstrip('0'))
        print(recapitulatif)

        plt.subplot(1, 2, 1)
        plt.plot(fauxPositifs[nom], vraisPositifs[nom], color=couleurs[i], label=nom + '(AUC = %0.4f)' % r_aucROC[nom])

        plt.subplot(1, 2, 2)
        plt.step(rec[nom], prec[nom], where='post', color=couleurs[i], label=f"{nom}(AP = {avgPrecRec[nom]:0.8f})")#alpha=0.8, 
        plt.fill_between(rec[nom], prec[nom], step='post', alpha=0.05)



    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Le taux de faux Positifs-(1 - Spécificité) = VN / (FP + VN)',size=18)
    plt.ylabel('Le taux de vrais positifs-Sensibilité = VP / (VP + FN)',size=18)
    plt.title('La courbe ROC (Receiver Operating Caracteristic)',size=20)
    plt.legend(loc="lower right")    

    plt.subplot(1, 2, 2)
    f_scores = np.linspace(0.2, 0.9, num=7)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Sensibilité(Rappel) = VP / (VP + FN)', size=18)
    plt.ylabel('Précision = VP / (VP + FP)', size=18)      
    plt.title('La courbe Précision-Rappel',size=20)
    plt.legend(loc="lower left")    

    projet.sauvegarderImage(f"Les courbes ROC et Précision-Rappel--{nom_essai}")    
    plt.show()
    
    #recapitulatif += 'Exécution  :'+('%.2fs' % (time.time() - t0)).lstrip('0')
    print('Exécution  :'+('%.2fs' % (time.time() - t0)).lstrip('0'))

    resultat = pd.DataFrame(pd.Series(r_aucROC), columns=["aucROC"])
    resultat['avgPrecRec']      = pd.Series(avgPrecRec   )
    resultat['accuracy']        = pd.Series(accuracy   )
    resultat['f1']              = pd.Series(f1         )
    resultat['precision']       = pd.Series(precision  )
    resultat['sensibilite']     = pd.Series(sensibilite)
    resultat['specificite']     = pd.Series(specificite)
    resultat['logloss']         = pd.Series(logloss    )
    resultat['hammingloss']     = pd.Series(hammingloss)
    resultat['jaccard']         = pd.Series(jaccard    )
    resultat["vrais_positifs"]  = pd.Series(vrais_positifs)
    resultat["vrais_negatifs"]  = pd.Series(vrais_negatifs)
    resultat["faux_positifs"]   = pd.Series(faux_positifs)
    resultat["faux_negatifs"]   = pd.Series(faux_negatifs)

    resultat['essai']           = nom_essai
    resultat                    = resultat.reset_index().rename(columns={'index':'Classifieur'}).set_index('Classifieur')
    
    resultat.sort_values('f1',ascending=False, inplace=True)
    return resultat

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