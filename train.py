import numpy as np
from embedding import Datasets
from model import CVAE
from tqdm import tqdm
import argparse
import os
import torch
from utils import to_device, idx2word, surface_realisation
from sklearn.metrics import normalized_mutual_info_score
import csv
from conversion import json2csv, csv2json

def anneal_fn(anneal_function, step, k, x, m):
    if anneal_function == 'logistic':
        return m*float(1/(1+np.exp(-k*(step-x))))
    elif anneal_function == 'linear':
        return m*min(1, step/x)
    
def loss_fn(logp, target, mean, logv, anneal_function, step, k1, x1, m1):
    
    target = target.view(-1)
    logp = logp.view(-1, logp.size(2))
    
    # Negative Log Likelihood
    NLL_loss = NLL(logp, target)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    KL_weight = anneal_fn(anneal_function, step, k1, x1, m1)

    return NLL_loss, KL_loss, KL_weight

def loss_labels(logc, target, anneal_function, step, k2, x2, m2):
    
    # Negative Log Likelihood
    label_loss = NLL(logc, target)
    label_weight = anneal_fn(anneal_function, step, k2, x2, m2)

    return label_loss, label_weight


def train(model, datasets, args):
    
    train_iter, val_iter = datasets.get_iterators(batch_size=args.batch_size)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    step = 0

    NLL_hist = []
    KL_hist = []
    NMI_hist = []
    acc_hist = []
    
    latent_rep={i:[] for i in range(args.n_classes)}
    
    for epoch in range(1, args.epochs + 1):
        tr_loss = 0.0
        NLL_tr_loss = 0.0
        KL_tr_loss = 0.0
        NMI_tr = 0.0
        acc_tr = 0.0
        
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
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, x, mean, logv,
                                                   args.anneal_function, step, args.k1, args.x1, args.m1)
            NLL_hist.append(NLL_loss/args.batch_size)
            KL_hist.append(KL_loss/args.batch_size)
            loss = (NLL_loss + KL_weight * KL_loss) #/args.batch_size

            if args.supervised:
                label_loss, label_weight = loss_labels(logc, y,
                                                       args.anneal_function, step, args.k2, args.x2, args.m2)
                loss += label_weight * label_loss
            else:
                entropy = torch.sum(c * torch.log(args.n_classes * c))
                loss += entropy

            pred_labels = logc.data.max(1)[1].long()
            acc = pred_labels.eq(y.data).cpu().sum().float()/args.batch_size
            acc_hist.append(acc)
            NMI = normalized_mutual_info_score(y.cpu().detach().numpy(), c.cpu().max(1)[1].numpy())
            NMI_hist.append(NMI)                
                
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            NLL_tr_loss += NLL_loss.item()
            KL_tr_loss += KL_loss.item()
            NMI_tr += NMI.item()
            acc_tr += acc.item()

        tr_loss     = tr_loss / len(datasets.train)
        NLL_tr_loss = NLL_tr_loss / len(datasets.train)
        KL_tr_loss  = KL_tr_loss / len(datasets.train)
        NMI_tr = NMI_tr / len(datasets.train)

        # calculate the validation loss for this epoch
        val_loss = 0.0
        NLL_val_loss = 0.0
        KL_val_loss = 0.0
        NMI_val = 0.0
        acc_val = 0.0
        
        model.eval() # turn on evaluation mode
        for batch in tqdm(val_iter): 
            x = getattr(batch, args.input_type)
            y = batch.intent
            x, y = to_device(x), to_device(y) 
            
            logp, mean, logv, logc, z = model(x)
            c = torch.exp(logc)
            
            # loss calculation
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, x, mean, logv,
                                                   args.anneal_function, step, args.k1, args.x1, args.m1)

            loss = (NLL_loss + KL_weight * KL_loss) #/args.batch_size

            if args.supervised:
                label_loss, label_weight = loss_labels(logc, y,
                                                       args.anneal_function, step, args.k2, args.x2, args.m2)
                loss += label_weight * label_loss
            else:
                entropy = torch.sum(c * torch.log(args.n_classes * c))
                loss += entropy

            pred_labels = logc.data.max(1)[1].long()             
            acc = pred_labels.eq(y.data).cpu().sum().float()/args.batch_size
            NMI = normalized_mutual_info_score(y.cpu().detach().numpy(), c.cpu().max(1)[1].numpy())

            val_loss += loss.item()
            NLL_val_loss += NLL_loss.item()
            KL_val_loss += KL_loss.item()
            NMI_val += NMI.item()
            acc_val += acc.item()
            
        val_loss     = val_loss / len(datasets.valid)
        NLL_val_loss = NLL_val_loss / len(datasets.valid)
        KL_val_loss  = KL_val_loss / len(datasets.valid)
        NMI_val = NMI_val / len(datasets.valid)
        
        print('Epoch {} : train {:.6f} valid {:.6f}'.format(epoch, tr_loss, val_loss))
        print('Training   :  NLL loss : {:.6f}, KL loss : {:.6f}, acc : {:.6f}, NMI : {:.6f}'.format(NLL_tr_loss, KL_tr_loss, acc_tr, NMI_tr))
        print('Validation :  NLL loss : {:.6f}, KL loss : {:.6f}, acc : {:.6f}, NMI : {:.6f}'.format(NLL_val_loss, KL_val_loss, acc_val, NMI_val))

    run['NLL_hist'] = NLL_hist
    run['KL_hist'] = KL_hist
    run['NMI_hist'] = NMI_hist
    run['acc_hist'] = acc_hist
    run['latent'] = latent_rep

    return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./data/train.csv')
    parser.add_argument('--validate_path', type=str, default='./data/validate.csv')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--save_model', type=str, default='model.pyT')
    parser.add_argument('--pickle', type=str, default='run.pyT')
    parser.add_argument('-spi', '--samples_per_intent', type=int, default=1000)
    parser.add_argument('--n_generated', type=int, default=100)
    parser.add_argument('--benchmark', action='store_true')

    parser.add_argument('--input_type', type=str, default='delexicalised', choices=['delexicalised', 'utterance'])
    parser.add_argument('--supervised', type=bool, default=True)
    parser.add_argument('--n_classes', type=int, default=7)
    parser.add_argument('-pr', '--print_reconstruction', type=int, default=-1, help='Print the reconstruction at a given epoch')

    parser.add_argument('--max_sequence_length', type=int, default=8)
    parser.add_argument('--emb_dim' , type=int, default=100)
    parser.add_argument('--tokenizer' , type=str, default='nltk', choices=['split', 'nltk', 'spacy'])
    parser.add_argument('--slot_averaging' , type=str, default='micro', choices=['none', 'micro', 'macro'])

    parser.add_argument('-ep', '--epochs', type=int, default=2)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)

    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru', choices=['rnn', 'gru', 'lstm'])
    parser.add_argument('-hs', '--hidden_size', type=int, default=64)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)

    parser.add_argument('-t', '--temperature', type=float, default=10)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.99)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic', choices=['logistic', 'linear'])
    parser.add_argument('-k1', '--k1', type=float, default=0.005)
    parser.add_argument('-x1', '--x1', type=int, default=100)
    parser.add_argument('-k2', '--k2', type=float, default=0.005)
    parser.add_argument('-x2', '--x2', type=int, default=50)
    parser.add_argument('-m1', '--m1', type=float, default=1.)
    parser.add_argument('-m2', '--m2', type=float, default=1.)

    run = {}

    args = parser.parse_args()
    run['args'] = args
    print(args)
    
    datadir = os.path.dirname(args.train_path)
    print('loading and embedding datasets')
    json2csv(datadir+'/2017-06-custom-intent-engines', datadir, samples_per_intent=args.samples_per_intent)
    datasets = Datasets(train_path=os.path.join(args.train_path), valid_path=os.path.join(args.validate_path), emb_dim=args.emb_dim, tokenizer=args.tokenizer)

    if args.input_type=='delexicalised':
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
            bidirectional=args.bidirectional,
            temperature=args.temperature
        )
        
    if args.load_model is not None:
        state_dict = torch.load(args.load_model)
        print(state_dict['embedding.weight'].size(), model.embedding.weight.size())
        if state_dict['embedding.weight'].size(0) != model.embedding.weight.size(0): # vocab changed
            state_dict['embedding.weight'] = vocab.vectors
            state_dict['outputs2vocab.weight'] = torch.randn(len(i2w), args.hidden_size*model.hidden_factor)
            state_dict['outputs2vocab.bias'] = torch.randn(len(i2w))
            
            print(state_dict['embedding.weight'].size(), model.embedding.weight.size())
        model.load_state_dict(state_dict)
    else:
        model.embedding.weight.data.copy_(vocab.vectors)

    model = to_device(model)
    print(model)
    
    train(model, datasets, args)
    if args.save_model is not None:
        torch.save(model.state_dict(), args.save_model)
    
    if args.n_generated>0:
    
        model.eval()

        samples, z, y_onehot = model.inference(n=args.n_generated)
        intent = y_onehot.data.max(1)[1].cpu().numpy()
        delexicalised = idx2word(samples, i2w=i2w, pad_idx=pad_idx)
        labelling, utterance = surface_realisation(samples, i2w=i2w, pad_idx=pad_idx)
        for i in range(args.n_generated):
            print('Intent : ', i2int[intent[i]])
            # print('Samples : ', samples[i])
            print('Delexicalised : ', delexicalised[i])
            print('Lexicalised : ', utterance[i] + '\n')

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

            run['metrics'] = {'raw':raw_metrics['average_metrics'], 'augmented':augmented_metrics['average_metrics'], 'improvement':{'intent':intent_improvement, 'slot':slot_improvement, 'score':score}}

    run['i2w'] = i2w
    run['w2i'] = w2i
    run['vectors'] = {'before':vocab.vectors, 'after':model.embedding.weight.data}
    
    torch.save(run, args.pickle)
