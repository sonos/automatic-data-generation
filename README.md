# automatic-data-generation

Automatic data generation with CVAEs -- Internship by Stéphane

## Install

Requirements: Python2.7, pip

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

## Training

Use the script `automatic-data-generation.train_and_eval_cvae.py` to train a model, generate sentences, and evaluate their quality.

```bash
python automatic_data_generation/train_and_eval_cvae.py --dataset-size 200 --n-generated 1000 --n-epochs 5 --none-size 100  --none-type subtitles --restrict-to-intent GetWeather PlayMusic
```

* `--dataset-size`: number of sentences in the training dataset
* `--none-size`: number of None sentences to be added to the training dataset 
* `--none-type`: type of None sentences
* `--restrict-to-intent`: list of intents to filter on for training
* `--n-epochs`: number of epochs for training
* `--n-generated`: number of generated sentences

## Output folder

An folder will be created with the following elements:

* `model`: a folder with a `model.pth` file and its associated `config.json`
* `tensorboard`: a folder with the checkpoints for tensorboard
* `run.pkl`: a dictionnary with every runtime parameters
* `train_*.csv`: the training dataset
* `train_*_augmented.csv`: the training dataset augmented with generated sentences
* `validate_*.csv`: the validation dataset

