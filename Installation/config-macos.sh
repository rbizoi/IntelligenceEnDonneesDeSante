#!/bin/bash
cd ~
pip install sql cx_Oracle ipython-sql

wget https://download.oracle.com/otn_software/mac/instantclient/198000/instantclient-basic-macos.x64-19.8.0.0.0dbru.zip
unzip instantclient-basic-macos.x64-19.8.0.0.0dbru.zip
ls ~/instantclient_19_8

cat <<FIN_FICHIER >> ~/.profile
LD_LIBRARY_PATH=~/instantclient_19_8:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
FIN_FICHIER


export PATH=~/instantclient_19_8:$PATH

export ARCHFLAGS="-arch x86_64"
python -m pip install cx_Oracle --upgrade --user

