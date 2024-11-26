# Bio-informatique et intelligence artificielle

<img src="https://raw.githubusercontent.com/rbizoi/IntelligenceEnDonneesDeSante/main/images/bioinformatique.png" width="512">

# Installation 

## Anaconda 

<img src="https://github.com/rbizoi/PythonFormation/blob/main/images/anaconda.png" width="512">


https://www.anaconda.com/download

## Environnement Python

```
conda activate root
conda info --envs
conda update -n base -c defaults conda
python -m pip install --upgrade pip
conda install scikit-learn-intelex

conda remove -n spark3 --all -y

conda create -n spark3 python==3.10 pyspark ipython ipython-sql jupyter notebook numpy pandas pyarrow scikit-image scikit-learn scikit-learn-intelex seaborn tifffile portpicker  biopython flatbuffers  redis colour pydot pygraphviz pyyaml imgaug tifffile imagecodecs lightgbm xgboost -y
conda activate spark3

pip install opencv-python kaggle pydot pydicom dash dash-bootstrap-components dash-core-components dash-html-components dash-table dash-draggable dash-cytoscape dash_daq Pillow jupyter_dash imagecorruptions imblearn sql sqlalchemy==1.4 psycopg2 openpyxl cx_oracle

```
