import torch

from automatic_data_generation.data.handlers.atis_dataset import AtisDataset
from automatic_data_generation.data.handlers.ptb_dataset import PTBDataset
from automatic_data_generation.data.handlers.snips_dataset import SnipsDataset
from automatic_data_generation.data.handlers.spam_dataset import SpamDataset
from automatic_data_generation.data.handlers.yelp_dataset import YelpDataset


def to_device(x, force_cpu):
    if torch.cuda.is_available() and not force_cpu:
        x = x.cuda()
    return x


def create_dataset(dataset_type, dataset_folder, input_type, dataset_size,
                   tokenizer_type, preprocessing_type, max_sequence_length,
                   embedding_type, embedding_dimension, max_vocab_size,
                   slot_averaging, run_dir, none_folder, none_idx, none_size):
    slotdic = None
    if dataset_type == "snips":
        dataset = SnipsDataset(
            dataset_folder=dataset_folder,
            input_type=input_type,
            dataset_size=dataset_size,
            tokenizer_type=tokenizer_type,
            preprocessing_type=preprocessing_type,
            max_sequence_length=max_sequence_length,
            embedding_type=embedding_type,
            embedding_dimension=embedding_dimension,
            max_vocab_size=max_vocab_size,
            output_folder=run_dir,
            none_folder=none_folder,
            none_idx=none_idx,
            none_size=none_size
        )
        if input_type == "delexicalised":
            dataset.embed_slots(slot_averaging)
            slotdic = dataset.get_slotdic()
    elif dataset_type == "atis":
        dataset = AtisDataset(
            dataset_folder=dataset_folder,
            input_type="utterance",
            dataset_size=dataset_size,
            tokenizer_type=tokenizer_type,
            preprocessing_type=preprocessing_type,
            max_sequence_length=max_sequence_length,
            embedding_type=embedding_type,
            embedding_dimension=embedding_dimension,
            max_vocab_size=max_vocab_size,
            output_folder=run_dir
        )
    elif dataset_type == "spam":
        dataset = SpamDataset(
            dataset_folder=dataset_folder,
            input_type="utterance",
            dataset_size=dataset_size,
            tokenizer_type=tokenizer_type,
            preprocessing_type=preprocessing_type,
            max_sequence_length=max_sequence_length,
            embedding_type=embedding_type,
            embedding_dimension=embedding_dimension,
            max_vocab_size=max_vocab_size,
            output_folder=run_dir
        )
    elif dataset_type == "yelp":
        dataset = YelpDataset(
            dataset_folder=dataset_folder,
            input_type="utterance",
            dataset_size=dataset_size,
            tokenizer_type=tokenizer_type,
            preprocessing_type=preprocessing_type,
            max_sequence_length=max_sequence_length,
            embedding_type=embedding_type,
            embedding_dimension=embedding_dimension,
            max_vocab_size=max_vocab_size,
            output_folder=run_dir
        )
    elif dataset_type == "penn-tree-bank":
        dataset = PTBDataset(
            dataset_folder=dataset_folder,
            input_type="utterance",
            dataset_size=dataset_size,
            tokenizer_type=tokenizer_type,
            preprocessing_type=preprocessing_type,
            max_sequence_length=max_sequence_length,
            embedding_type=embedding_type,
            embedding_dimension=embedding_dimension,
            max_vocab_size=max_vocab_size,
            output_folder=run_dir
        )
    else:
        raise TypeError("Unknown dataset type")

    return dataset, slotdic
