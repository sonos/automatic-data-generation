import numpy as np
from embedding import Datasets
from model import VAE, CVAE
from tqdm import tqdm
import argparse
import os
import torch
from utils import to_device, idx2word, surface_realisation
from sklearn.metrics import normalized_mutual_info_score
import csv
from conversion import csv2json

def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)
    
def loss_fn(logp, target, mean, logv, anneal_function, step, k, x0):
    
    target = target.view(-1)
    logp = logp.view(-1, logp.size(2))
    
    # Negative Log Likelihood
    NLL_loss = NLL(logp, target)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    KL_weight = kl_anneal_function(anneal_function, step, k, x0)

    return NLL_loss, KL_loss, KL_weight

def train(model, datasets, args):
    
    train_iter, val_iter = datasets.get_iterators(batch_size=args.batch_size)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    step = 0

    NLL_hist = []
    KL_hist = []
    NMI_hist = []

    latent_rep={i:[] for i in range(args.n_classes)}
    
    for epoch in range(1, args.epochs + 1):
        tr_loss = 0.0
        NLL_tr_loss = 0.0
        KL_tr_loss = 0.0
        NMI_tr = 0.0
        
        model.train() # turn on training mode
        for batch in tqdm(train_iter): 
            step += 1
            opt.zero_grad()

            x = getattr(batch, args.input_type)
            y = batch.intent
            x, y = to_device(x), to_device(y)
            
            logp, mean, logv, logc, z = model(x)
            for i,intent in enumerate(y):
                latent_rep[int(intent)].append(z[i].cpu().detach().numpy())
            c = torch.exp(logc)

            # to inspect input and output
            if epoch == args.print_reconstruction:
                x_sentences = x.transpose(0,1)[:3]
                print('\nInput sentences :')
                print(*idx2word(x_sentences, i2w=i2w, pad_idx=pad_idx), sep='\n')
                _, y_sentences = torch.topk(logp, 1, dim=-1)
                y_sentences = y_sentences.transpose(0,1)[:3]
                print('\nOutput sentences : ')
                print(*idx2word(y_sentences, i2w=i2w, pad_idx=pad_idx), sep='\n')
                print('\n')
            
            # loss calculation
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, x,
                    mean, logv, args.anneal_function, step, args.k, args.x0)
            NLL_hist.append(NLL_loss/args.batch_size)
            KL_hist.append(KL_loss/args.batch_size)
            loss = (NLL_loss + KL_weight * KL_loss) #/args.batch_size

            if args.conditional:
                entropy = torch.sum(c * torch.log(args.n_classes * c))
                NMI = normalized_mutual_info_score(y.cpu().detach().numpy(), c.cpu().max(1)[1].numpy())
                NMI_hist.append(NMI)
                loss += entropy
                
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            NLL_tr_loss += NLL_loss.item()
            KL_tr_loss += KL_loss.item()
            NMI_tr += NMI.item()

        tr_loss     = tr_loss / len(datasets.train)
        NLL_tr_loss = NLL_tr_loss / len(datasets.train)
        KL_tr_loss  = KL_tr_loss / len(datasets.train)
        NMI_tr = NMI_tr / len(datasets.train)

        # calculate the validation loss for this epoch
        val_loss = 0.0
        NLL_val_loss = 0.0
        KL_val_loss = 0.0
        NMI_val = 0.0

        model.eval() # turn on evaluation mode
        for batch in tqdm(val_iter): 
            x = getattr(batch, args.input_type)
            y = batch.intent
            x, y = to_device(x), to_device(y) 
            
            logp, mean, logv, logc, z = model(x)
            c = torch.exp(logc)
            
            # loss calculation
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, x,
                    mean, logv, args.anneal_function, step, args.k, args.x0)

            loss = (NLL_loss + KL_weight * KL_loss) #/args.batch_size

            if args.conditional:
                entropy = torch.sum(c * torch.log(args.n_classes * c))
                NMI = normalized_mutual_info_score(y.cpu().detach().numpy(), c.cpu().max(1)[1].numpy())
                loss += entropy

            val_loss += loss.item()
            NLL_val_loss += NLL_loss.item()
            KL_val_loss += KL_loss.item()
            NMI_val += NMI.item()

        val_loss     = val_loss / len(datasets.valid)
        NLL_val_loss = NLL_val_loss / len(datasets.valid)
        KL_val_loss  = KL_val_loss / len(datasets.valid)
        NMI_val  = NMI_val / len(datasets.valid)
        
        print('Epoch {} : train {:.6f} valid {:.6f}'.format(epoch, tr_loss, val_loss))
        print('Training   :  NLL loss : {:.6f}, KL loss : {:.6f}, NMI : {:.6f}'.format(NLL_tr_loss, KL_tr_loss, NMI_tr))
        print('Validation :  NLL loss : {:.6f}, KL loss : {:.6f}, NMI : {:.6f}'.format(NLL_val_loss, KL_val_loss, NMI_val))

    run['NLL_hist'] = NLL_hist
    run['KL_hist'] = KL_hist
    run['NMI_hist'] = NMI_hist
    run['latent'] = latent_rep

    return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./data/train.csv')
    parser.add_argument('--validate_path', type=str, default='./data/validate.csv')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--save_model', type=str, default='model.pyT')
    parser.add_argument('--pickle', type=str, default='run.pyT')
    parser.add_argument('--n_generated', type=int, default=100)
    parser.add_argument('--benchmark', action='store_true')

    parser.add_argument('--input_type', type=str, default='delexicalised', choices=['delexicalised', 'utterance'])
    parser.add_argument('--conditional', type=int, default=1)
    parser.add_argument('--n_classes', type=int, default=7)
    parser.add_argument('-pr', '--print_reconstruction', type=int, default=-1, help='Print the reconstruction at a given epoch')

    parser.add_argument('--max_sequence_length', type=int, default=10)
    parser.add_argument('--emb_dim' , type=int, default=100)
    parser.add_argument('--tokenizer' , type=str, default='nltk', choices=['split', 'nltk', 'spacy'])
    parser.add_argument('--slot_averaging' , type=str, default='micro', choices=['none', 'micro', 'macro'])

    parser.add_argument('-ep', '--epochs', type=int, default=2)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)

    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru', choices=['rnn', 'gru', 'lstm'])
    parser.add_argument('-hs', '--hidden_size', type=int, default=512)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.9)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.2)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic', choices=['logistic', 'linear'])
    parser.add_argument('-k', '--k', type=float, default=0.01)
    parser.add_argument('-x0', '--x0', type=int, default=500)

    run = {}

    args = parser.parse_args()
    run['args'] = args
    print(args)
    
    print('loading and embedding datasets')
    datasets = Datasets(train_path=os.path.join(args.train_path), valid_path=os.path.join(args.validate_path), emb_dim=args.emb_dim, tokenizer=args.tokenizer)

    if args.input_type=='utterance':
        print('embedding the slots with %s averaging' %args.slot_averaging)
        datasets.embed_slots(args.slot_averaging)
    
    vocab = datasets.TEXT.vocab if args.input_type=='utterance' else datasets.DELEX.vocab
    i2w = vocab.itos
    w2i = vocab.stoi
    i2int = datasets.INTENT.vocab.itos
    int2i = datasets.INTENT.vocab.stoi
    sos_idx = w2i['SOS']
    eos_idx = w2i['EOS']
    pad_idx = w2i['<pad>']
    unk_idx = w2i['<unk>']

    NLL = torch.nn.NLLLoss(reduction='sum', ignore_index=pad_idx)

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
            n_classes=args.n_classes,
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

    torch.save(run, args.pickle)
    
    if args.n_generated>0:
    
        model.eval()

        samples, z, y_onehot = model.inference(n=args.n_generated)
        intent = y_onehot.data.max(1)[1].cpu().numpy()
        print('----------SAMPLES----------')
        print(samples)
        delexicalised = idx2word(samples, i2w=i2w, pad_idx=pad_idx)
        print('----------DELEXICALISED----------')
        print(*delexicalised, sep='\n')
        labelling, utterance = surface_realisation(samples, i2w=i2w, pad_idx=pad_idx)
        print('----------LEXICALISED----------')
        print(*utterance, sep='\n')
        print('----------INTENTS----------')
        print(*[i2int[int] for int in intent], sep='\n')

        run['generated'] = utterance
        
        augmented_path = args.train_path.replace('.csv', '_augmented.csv')
        print('Dumping augmented dataset at %s' %augmented_path)
        from shutil import copyfile
        copyfile(args.train_path, augmented_path)
        csvfile    = open(augmented_path, 'a')
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for u, l, d, i in zip(utterance, labelling, delexicalised, intent):
            csv_writer.writerow([u, l, d, i2int[i]])

        if args.benchmark:
            from snips_nlu import SnipsNLUEngine
            from snips_nlu_metrics import compute_train_test_metrics

            datadir = os.path.join(*args.train_path.split('/')[:-1])
            csv2json(datadir, datadir, augmented=False)
            csv2json(datadir, datadir, augmented=True)

            print('Starting benchmarking...')

            def my_matching_lambda(lhs_slot, rhs_slot):
                return lhs_slot['text'].strip() == rhs_slot["rawValue"].strip()

            raw_metrics = compute_train_test_metrics(train_dataset="data/train.json",
                                                    test_dataset="data/validate.json",
                                                    engine_class=SnipsNLUEngine,
                                                    slot_matching_lambda = my_matching_lambda
                                                    )
            augmented_metrics = compute_train_test_metrics(train_dataset="data/train_augmented.json",
                                                    test_dataset="data/validate.json",
                                                    engine_class=SnipsNLUEngine,
                                                    slot_matching_lambda = my_matching_lambda
                                                    )

            print('----------METRICS----------')
            print('Without augmentation : ')
            print(raw_metrics['average_metrics'])
            print('With augmentation : ')
            print(augmented_metrics['average_metrics'])
            intent_improvement = 100 * ((augmented_metrics['average_metrics']['intent']['f1'] - raw_metrics['average_metrics']['intent']['f1'])
                                        / raw_metrics['average_metrics']['intent']['f1'])
            slot_improvement = 100 * ((augmented_metrics['average_metrics']['slot']['f1'] - raw_metrics['average_metrics']['slot']['f1'])
                                        / raw_metrics['average_metrics']['slot']['f1'])
            score = intent_improvement + slot_improvement

            print('Improvement metrics : intent {:.4f} slot {:.4f} total {:.4f}'.format(intent_improvement, slot_improvement, score))

            run['metrics'] = {'raw':raw_metrics, 'augmented':augmented_metrics}
