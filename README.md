# Bio-informatique et intelligence artificielle

<img src="https://raw.githubusercontent.com/rbizoi/IntelligenceEnDonneesDeSante/main/images/bioinformatique.png" width="512">

# Installation 

## Anaconda 

<img src="https://raw.githubusercontent.com/rbizoi/IntelligenceEnDonneesDeSante/refs/heads/main/images/anaconda.png" width="512">


https://www.anaconda.com/download/success

## Environnement Python
Mise à jour des librairies de l’environnement base
```
conda activate root
conda update --all
python -m pip install --upgrade pip
```

### Windows
```
conda create -n cours -c conda-forge  python==3.12 ipython ipython-sql jupyter notebook numpy pandas pyarrow matplotlib seaborn portpicker biopython flatbuffers redis colour pydot pygraphviz pyyaml pyspark folium scikit-image scikit-learn yellowbrick lightgbm xgboost catboost plotly imgaug tifffile imagecodecs kneed imbalanced-learn 

conda activate cours
#conda remove -n cours --all -y

#pip install ipython-sql sql psycopg2
#pip uninstall matplotlib seaborn
#pip install matplotlib seaborn opencv-python-headless
```

### Linux

```
conda create -p /home/utilisateur/anaconda3/envs/cours -c conda-forge  python==3.12 ipython ipython-sql jupyter notebook numpy pandas pyarrow matplotlib seaborn portpicker biopython flatbuffers redis colour pydot pygraphviz pyyaml pyspark folium scikit-image scikit-learn yellowbrick lightgbm xgboost catboost plotly imgaug tifffile imagecodecs kneed imbalanced-learn
conda activate cours
# conda remove -n cours --all -y

#pip install ipython-sql sql psycopg2
#pip uninstall matplotlib seaborn
#pip install matplotlib seaborn opencv-python-headless
```

