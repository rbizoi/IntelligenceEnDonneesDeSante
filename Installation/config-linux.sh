#!/bin/bash
cd ~
pip install sql cx_Oracle ipython-sql

wget https://download.oracle.com/otn_software/linux/instantclient/214000/instantclient-basic-linux.x64-21.4.0.0.0dbru.zip
unzip instantclient-basic-linux.x64-21.4.0.0.0dbru.zip
ls ~/instantclient_21_4

cat <<FIN_FICHIER >> ~/.profile
export LD_LIBRARY_PATH=~/instantclient_21_4:\$LD_LIBRARY_PATH
FIN_FICHIER
