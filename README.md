# Bio-informatique et intelligence artificielle

<img src="https://raw.githubusercontent.com/rbizoi/IntelligenceEnDonneesDeSante/main/images/bioinformatique.png" width="512">

# Installation 

## Anaconda 

<img src="https://raw.githubusercontent.com/rbizoi/IntelligenceEnDonneesDeSante/refs/heads/main/images/anaconda.png" width="512">


https://www.anaconda.com/download/success

## Environnement Python

### Windows
```
conda create -n cours python==3.10 ipython ipython-sql jupyter notebook numpy==1.23.5 pandas pyyaml==5.4.1 pyarrow scikit-image scikit-learn matplotlib seaborn  tifffile portpicker biopython Flask==2.0.2 Flask-Caching==1.10.1 Flask-Compress==1.10.1 flatbuffers  redis colour pydot pygraphviz pyyaml imgaug tifffile imagecodecs pyspark sqlalchemy

conda activate cours
# conda remove -n cours --all -y

pip install ipython-sql sql psycopg2
pip uninstall matplotlib seaborn
pip install matplotlib seaborn opencv-python-headless
```

### Linux

```
conda create -p /home/utilisateur/anaconda3/envs/cours python==3.10 ipython ipython-sql jupyter notebook numpy==1.23.5 pandas pyyaml==5.4.1 pyarrow scikit-image scikit-learn matplotlib seaborn  tifffile portpicker biopython Flask==2.0.2 Flask-Caching==1.10.1 Flask-Compress==1.10.1 flatbuffers  redis colour pydot pygraphviz pyyaml imgaug tifffile imagecodecs pyspark sqlalchemy

conda activate cours
# conda remove -n cours --all -y

pip install ipython-sql sql psycopg2
pip uninstall matplotlib seaborn
pip install matplotlib seaborn opencv-python-headless
```

