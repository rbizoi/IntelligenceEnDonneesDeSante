# Création environnement Python

https://www.anaconda.com/download

```shell
# windows 
conda create -p coursDB ipython jupyter notebook numpy pandas pyarrow seaborn \
               sqlalchemy ipython-sql psycopg2 openpyxl
               
# linux               
conda create -p /home/utilisateur/anaconda3/envs/coursDB ipython jupyter notebook numpy pandas pyarrow seaborn \
               sqlalchemy ipython-sql psycopg2 openpyxl
               
conda activate coursDB

```