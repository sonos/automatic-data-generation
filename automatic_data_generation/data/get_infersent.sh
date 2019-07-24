#!/bin/bash
# This script will download everything required to make InferSent work, as used by embed_intents.py

git clone https://github.com/facebookresearch/InferSent.git
cd InferSent
mkdir GloVe
curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip GloVe/glove.840B.300d.zip -d GloVe/
mkdir encoder
curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
