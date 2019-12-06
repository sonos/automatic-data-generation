# automatic-data-generation

Pytorch implementation of experiments described in "Conditioned Query Generation for Task-Oriented Dialogue Systems".
This is a work in progress, feel free to reach out for any question.

## Install

Requirements: Python3.6, pip

```
virtualenv venv
. venv/bin/activate
pip install -e .
```

You might need to download some NLTK resources:

```
  >>> import nltk
  >>> nltk.download('punkt')
```

## Dataset

The abstract class `automatic-data-generation.data.base_dataset.py` provides the interface for representing a training dataset. To implement a new dataset format, write a class inheriting from Dataset and implement its abstract methods.

You then need to allow for a new `dataset_type` in the training script 
`automatic_data_generation/train_and_eval_cvae.py` which should be the name 
of the sub-directory in your data folder.

Finally, you should update the data factory `create_dataset` in
`automatic_data_generation/utils/utils.py`.

## None sentences

The reservoir dataset of unannotated queries used for transfer experiments 
in the paper is not publicly available. To explore the query transfer 
method, you need to add you own None sentences in csv format in a 
sub-directory in your data folder. 

You need to first download the InferSent model by running the following 
executable:
```bash
./automatic_data_generation/data/get_infersent.sh
```

To embed your None sentences, run the following command:
```bash
python automatic_data_generation/data/utils/embed_intents.py --dataset_path 
./your/none/data/path
```

You then need to allow for a new `none_type` in the training script 
`automatic_data_generation/train_and_eval_cvae.py` which should be the name 
of the sub-directory in your data folder.

Add the index of the utterances in the csv file to the 
`NONE_COLUMN_MAPPING` dictionary in 
`automatic_data_generation/data/utils/utils.py`. If the utterance is the 
first (resp. n) field of the csv, add 0 (resp. n+1).

You should be good to go.

## Training

Use the script `automatic-data-generation.train_and_eval_cvae.py` to train a model, generate sentences, and evaluate their quality.

For a simple run without query transfer, you may run:
```bash
python automatic_data_generation/train_and_eval_cvae.py -ep 10 --n-generated 100 --dataset-size 125
```

Possible options are:
* `--dataset-size`: number of sentences in the training dataset
* `--none-size`: number of None sentences to be added to the training dataset 
* `--none-type`: type of None sentences
* `--restrict-to-intent`: list of intents to filter on for training
* `--n-epochs`: number of epochs for training
* `--n-generated`: number of generated sentences
* `--infersent-selection`: possible query transfer schemes, `unsupervised` 
is the normal scheme, `supervised` is 
the pseudolabelling baseline, and `NO_INFERSENT_SELECTION` deactivates the 
feature
* `--cosine-thresholds`: the selection threshold for query transfer (defaults
 to 0.9)
* `alpha`: the parameter regulating transfer


If you have added your own None type, a typical run may be:
```bash
python automatic_data_generation/train_and_eval_cvae.py -ep 50 --n-generated
 2000 --dataset-size 125 --none-size 125 --none-type mynonetype 
 --infersent-selection unsupervised --cosine-threshold 0.9 --alpha:0.1
```
                   
## Output folder

An folder will be created with the following elements:
* `load`: a folder with a `model.pth` file and its associated `config.json` 
and a `vocab.pth` file containing the vocabulary
* `tensorboard`: a folder with the checkpoints for tensorboard
* `run.pkl`: a dictionnary with every runtime parameters
* `train_*.csv`: the training dataset
* `train_*_augmented.csv`: the training dataset augmented with generated sentences
* `validate_*.csv`: the validation dataset

