import random
import torch

from automatic_data_generation.data.handlers.atis_dataset import AtisDataset
from automatic_data_generation.data.handlers.ptb_dataset import PTBDataset
from automatic_data_generation.data.handlers.snips_dataset import SnipsDataset
from automatic_data_generation.data.handlers.yelp_dataset import YelpDataset


def to_device(x, force_cpu):
    if torch.cuda.is_available() and not force_cpu:
        x = x.cuda()
    return x


def create_dataset(dataset_type,
                   dataset_folder, dataset_size, restrict_intents,
                   none_folder, none_size, none_intents, none_idx,
                   infersent_selection, cosine_threshold,
                   input_type, tokenizer_type,
                   preprocessing_type, max_sequence_length,
                   embedding_type, embedding_dimension, max_vocab_size,
                   slot_embedding, run_dir):
    
    if dataset_type.startswith("snips"):
        dataset = SnipsDataset(
            dataset_folder=dataset_folder,
            dataset_size=dataset_size,
            restrict_intents=restrict_intents,
            none_folder=none_folder,
            none_size=none_size,
            none_intents=none_intents,
            none_idx=none_idx,
            infersent_selection=infersent_selection,
            cosine_threshold=cosine_threshold,
            input_type=input_type,
            tokenizer_type=tokenizer_type,
            preprocessing_type=preprocessing_type,
            max_sequence_length=max_sequence_length,
            embedding_type=embedding_type,
            embedding_dimension=embedding_dimension,
            max_vocab_size=max_vocab_size,
            output_folder=run_dir
        )

    elif dataset_type == "atis":
        dataset = AtisDataset(
            dataset_folder=dataset_folder,
            dataset_size=dataset_size,
            restrict_intents=restrict_intents,
            none_folder=none_folder,
            none_size=none_size,
            none_intents=none_intents,
            none_idx=none_idx,
            infersent_selection=infersent_selection,
            cosine_threshold=cosine_threshold,
            input_type=input_type,
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
            dataset_size=dataset_size,
            restrict_intents=restrict_intents,
            none_folder=none_folder,
            none_size=none_size,
            none_intents=none_intents,
            none_idx=none_idx,
            infersent_selection=infersent_selection,
            cosine_threshold=cosine_threshold,
            input_type=input_type,
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
            dataset_size=dataset_size,
            restrict_intents=restrict_intents,
            none_folder=none_folder,
            none_size=none_size,
            none_intents=none_intents,
            none_idx=none_idx,
            infersent_selection=infersent_selection,
            cosine_threshold=cosine_threshold,
            input_type=input_type,
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

    dataset.embed_unks(num_special_toks=4)
    if input_type == "delexicalised":
        dataset.build_slotdic()
        dataset.embed_slots(slot_embedding, dataset.slotdic)

    return dataset
