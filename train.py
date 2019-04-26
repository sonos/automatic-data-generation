import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from embedding import Datasets
from model import VAE, CVAE
from tqdm import tqdm
import argparse
import os
import torch
from utils import to_device, idx2word

def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

def loss_fn(logp, target, mean, logv, anneal_function, step, k, x0):
    
    target = target.view(-1)
    logp = logp.view(-1, logp.size(2))
    
    # Negative Log Likelihood
    NLL_loss = F.nll_loss(logp, target)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    KL_weight = kl_anneal_function(anneal_function, step, k, x0)

    return NLL_loss, KL_loss, KL_weight

def train(model, datasets, args):
    
    train_iter, val_iter = datasets.get_iterators(batch_size=args.batch_size)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    step = 0
    
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        NLL_running_loss = 0.0
        KL_running_loss = 0.0
        
        model.train() # turn on training mode
        for batch in tqdm(train_iter): 
            step += 1
            opt.zero_grad()

            x = getattr(batch, args.input_type)
            y = batch.intent - 1
            x, y = to_device(x), to_device(y) 
            
            logp, mean, logv, z = model(x)
            
            # loss calculation
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, x,
                    mean, logv, args.anneal_function, step, args.k, args.x0)

            loss = (NLL_loss + KL_weight * KL_loss) #/args.batch_size

            loss.backward()
            opt.step()

            running_loss += loss.item()
            NLL_running_loss += NLL_loss.item()
            KL_running_loss += KL_loss.item()

        epoch_loss     = running_loss / len(datasets.train)
        NLL_epoch_loss = NLL_running_loss / len(datasets.train)
        KL_epoch_loss  = KL_running_loss / len(datasets.train)
        

        # calculate the validation loss for this epoch
        val_loss = 0.0
        NLL_val_loss = 0.0
        KL_val_loss = 0.0

        model.eval() # turn on evaluation mode
        for batch in tqdm(val_iter): 
            x = getattr(batch, args.input_type)
            y = batch.intent - 1
            x, y = to_device(x), to_device(y) 
            
            logp, mean, logv, z = model(x)
            
            # loss calculation
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, x,
                    mean, logv, args.anneal_function, step, args.k, args.x0)

            loss = (NLL_loss + KL_weight * KL_loss) #/args.batch_size

            val_loss += loss.item()
            NLL_val_loss += NLL_loss.item()
            KL_val_loss += KL_loss.item()


        val_loss     = val_loss / len(datasets.valid)
        NLL_val_loss = NLL_val_loss / len(datasets.valid)
        KL_val_loss  = KL_val_loss / len(datasets.valid)
        
        print('Epoch: {}'.format(epoch, epoch_loss, val_loss))
        print('Training :  NLL loss : {:.6f}, KL loss : {:.6f}'.format(NLL_epoch_loss, KL_epoch_loss))
        print('Valid    :  NLL loss : {:.6f}, KL loss : {:.6f}'.format(NLL_val_loss, KL_val_loss))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--save_model', type=str, default=None)
    parser.add_argument('--n_generated', type=int, default=5)
    parser.add_argument('--input_type', type=str, default='delexicalised', choices=['delexicalised', 'utterance'])
    parser.add_argument('--conditional', type=int, default=1)

    parser.add_argument('--max_sequence_length', type=int, default=10)
    parser.add_argument('--emb_dim' , type=int, default=100)
    parser.add_argument('--tokenizer' , type=str, default='split', choices=['split', 'nltk', 'spacy'])
    parser.add_argument('--slot_averaging' , type=str, default='micro', choices=['none', 'micro', 'macro'])

    parser.add_argument('-ep', '--epochs', type=int, default=2)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)

    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru', choices=['rnn', 'gru', 'lstm'])
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic', choices=['logistic', 'linear'])
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    args = parser.parse_args()
    
    print(args)
    
    print('loading and embedding datasets')
    datasets = Datasets(train_path=os.path.join(args.datadir,'train.csv'), valid_path=os.path.join(args.datadir, 'validate.csv'), emb_dim=args.emb_dim, tokenizer='split')

    if args.input_type=='utterance':
        print('embedding the slots with %s averaging' %args.slot_averaging)
        datasets.embed_slots(args.slot_averaging)
    
    vocab = datasets.TEXT.vocab if args.input_type=='utterance' else datasets.DELEX.vocab
    i2w = vocab.itos
    w2i = vocab.stoi
    sos_idx = w2i['#']
    eos_idx = w2i['.']
    pad_idx = w2i['<pad>']
    unk_idx = w2i['<unk>']
    
    NLL = torch.nn.NLLLoss(size_average=False, ignore_index=pad_idx)

    if args.conditional:
        model = CVAE(
            vocab_size=len(i2w),
            max_sequence_length=args.max_sequence_length,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            unk_idx=unk_idx,
            embedding_size=args.emb_dim,
            rnn_type=args.rnn_type,
            hidden_size=args.hidden_size,
            word_dropout=args.word_dropout,
            embedding_dropout=args.embedding_dropout,
            z_size=args.latent_size,
            n_classes=7,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional
        )
    else:
        model = VAE(
            vocab_size=len(i2w),
            max_sequence_length=args.max_sequence_length,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            unk_idx=unk_idx,
            embedding_size=args.emb_dim,
            rnn_type=args.rnn_type,
            hidden_size=args.hidden_size,
            word_dropout=args.word_dropout,
            embedding_dropout=args.embedding_dropout,
            latent_size=args.latent_size,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional
        )
        
    model.embedding.weight.data.copy_(vocab.vectors)
    model = to_device(model)
    print(model)
    
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model))
    
    train(model, datasets, args)
    
    if args.save_model is not None:
        torch.save(model.state_dict(), args.save_model)
        
    if args.n_generated>0:
    
        model.eval()

        samples, z = model.inference(n=args.n_generated)
        print('----------SAMPLES----------')
        print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

