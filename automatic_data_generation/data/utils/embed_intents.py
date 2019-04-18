import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from automatic_data_generation.utils.io import read_csv


def embed_dataset(dataset_path, infersent_path, force_cpu=False):
    """
    To make this work, first run ./get_infersent.sh
    """
    MODEL_PATH = infersent_path / "encoder/infersent1.pkl"
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
    model = InferSent(params_model)
    if force_cpu:
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.cuda()

    W2V_PATH = infersent_path / 'GloVe/glove.840B.300d.txt'
    model.set_w2v_path(W2V_PATH)
    model.build_vocab_k_words(K=100000)

    csv_data = read_csv(dataset_path / 'train.csv')
    csv_data = csv_data[1:]  # skip header
    data = defaultdict(list)

    for irow, row in enumerate(csv_data):
        if 'snips' in str(dataset_path):
            utterance, labels, delexicalised, intent = row
        else:
            raise TypeError(
                "Unknown dataset type. Implement your own first. See the "
                "README")
        data[intent].append(utterance)

    vectors = {}
    for i, (intent, sentences) in enumerate(data.items()):
        print('{}/{} done'.format(i, len(data.items())))
        embeddings = model.encode(sentences)
        avg_embedding = np.mean(embeddings, axis=0)
        vectors[intent] = avg_embedding

    return vectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='data/snips/')
    parser.add_argument('--infersent_path', type=str, default='InferSent/')
    parser.add_argument('--force_cpu', action='store_true', default=False)

    args = parser.parse_args()

    infersent_path = Path(args.infersent_path)
    dataset_path = Path(args.dataset_path)

    sys.path.append(str(infersent_path))
    from models import InferSent

    vectors = embed_dataset(dataset_path, infersent_path, args.force_cpu)
    torch.save(vectors, dataset_path / 'vectors.pkl')
