import numpy as np
from automatic_data_generation.models.embedding import Datasets
import automatic_data_generation.models.cvae as models
from tqdm import tqdm
import argparse
import os
import torch
from automatic_data_generation.utils.utils import to_device, idx2word, surface_realisation
from automatic_data_generation.utils.metrics import calc_bleu, calc_entropy, calc_diversity, intent_classification
from sklearn.metrics import normalized_mutual_info_score
import csv
from automatic_data_generation.utils.conversion import csv2json
import ipdb

def anneal_fn(anneal_function, step, k, x, m):
    if anneal_function == 'logistic':
        return m*float(1/(1+np.exp(-k*(step-x))))
    elif anneal_function == 'linear':
        return m*min(1, step/x)


def loss_fn(logp, bow, target, length, mean, logv, anneal_function, step, k1, x1, m1):

    batch_size, seqlen, vocab_size = logp.size()
    target = target.view(batch_size, -1)

    # Bag of words
    if bow is not None:
        bow.view(batch_size,-1)
        BOW_loss = - torch.einsum('iik->', bow[:,target])
    else:
        BOW_loss = torch.Tensor([0])
        
    # cut-off unnecessary padding from target, and flatten
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp.view(-1, vocab_size)
    # Negative Log Likelihood
    NLL_loss = NLL_recon(logp, target)

    # KL Divergence
    KL_losses = -0.5 * torch.sum((1 + logv - mean.pow(2) - logv.exp()), dim=0)
    KL_weight = anneal_fn(anneal_function, step, k1, x1, m1)
    
    return NLL_loss, KL_losses, KL_weight, BOW_loss


def loss_labels(logc, target, anneal_function, step, k2, x2, m2):
    
    # Negative Log Likelihood
    label_loss = NLL_label(logc, target)
    label_weight = anneal_fn(anneal_function, step, k2, x2, m2)

    return label_loss, label_weight


def train(model, datasets, args):
    
    train_iter, val_iter = datasets.get_iterators(batch_size=args.batch_size)

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # opt = torch.optim.Adam([
    #     {"params": model.encoder_rnn.parameters(), "lr": args.learning_rate},
    #     {"params": model.hidden2mean.parameters(), "lr": args.learning_rate},
    #     {"params": model.hidden2logv.parameters(), "lr": args.learning_rate},
    #     {"params": model.hidden2cat.parameters(),  "lr": args.learning_rate},
    #     {"params": model.latent2hidden.parameters(), "lr": args.learning_rate},
    #     {"params": model.latent2bow.parameters(), "lr": args.learning_rate},
    #     {"params": model.outputs2vocab.parameters(), "lr": args.learning_rate}])

    step = 0

    NLL_hist = []
    KL_hist = []
    BOW_hist = []
    NMI_hist = []
    acc_hist = []
    
    latent_rep={i:[] for i in range(model.n_classes)}
    
    for epoch in range(1, args.epochs + 1):
        tr_loss = 0.0
        NLL_tr_loss = 0.0
        KL_tr_loss = 0.0
        BOW_tr_loss = 0.0
        NMI_tr = 0.0
        n_correct_tr = 0.0
        acc_tr = 0.0
        
        model.train() # turn on training mode
        for iteration, batch in enumerate(tqdm(train_iter)): 
            step += 1
            opt.zero_grad()
            # model.word_dropout_rate =  anneal_fn(args.anneal_function, step, args.k3, args.x3, args.m3)

            x, lengths = getattr(batch, args.input_type)
            input = x[:, :-1] # remove <eos>
            target = x[:, 1:] # remove <sos>
            lengths -= 1 # account for the removal
            input, target = to_device(input), to_device(target)
            if args.conditional != 'none':
                y = batch.intent.squeeze()
                y = to_device(y)
                sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
                y = y[sorted_idx]

            logp, mean, logv, logc, z, bow = model(input, lengths)
            if epoch == args.epochs and args.conditional != 'none':
                for i,intent in enumerate(y):
                    latent_rep[int(intent)].append(z[i].cpu().detach().numpy())
            
            # loss calculation
            NLL_loss, KL_losses, KL_weight, BOW_loss = loss_fn(logp, bow, target, lengths, mean, logv,
                                                   args.anneal_function, step, args.k1, args.x1, args.m1)
            KL_loss = torch.sum(KL_losses)
            NLL_hist.append(NLL_loss.detach().cpu().numpy()/args.batch_size)
            KL_hist.append(KL_losses.detach().cpu().numpy()/args.batch_size)
            BOW_hist.append(BOW_loss.detach().cpu().numpy()/args.batch_size)
            label_loss, label_weight = loss_labels(logc, y,
                                                   args.anneal_function, step, args.k2, args.x2, args.m2)
            loss = (NLL_loss + KL_weight * KL_loss + label_weight * label_loss) #/args.batch_size

            if args.bow_loss:
                loss += BOW_loss
                
            if args.conditional=='none':
                pred_labels = 0
                n_correct = 0
                NMI = 0
            else:
                if args.conditional=='supervised':
                    label_loss, label_weight = loss_labels(logc, y,
                                                       args.anneal_function, step, args.k2, args.x2, args.m2)
                    loss += label_weight * label_loss
                elif args.conditional=='unsupervised':
                    entropy = torch.sum(torch.exp(logc) * torch.log(model.n_classes * torch.exp(logc)))
                    loss += entropy
                pred_labels = logc.data.max(1)[1].long()
                n_correct = pred_labels.eq(y.data).cpu().sum().float().item()
                acc_hist.append(n_correct/args.batch_size)
                NMI = normalized_mutual_info_score(y.cpu().detach().numpy(), torch.exp(logc).cpu().max(1)[1].numpy())
                NMI_hist.append(NMI)                
                
            loss.backward()
            # CLIPPING
            # for p in model.parameters():
            #     p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
            # torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value=1)
            opt.step()

            tr_loss += loss.item()
            NLL_tr_loss += NLL_loss.item()
            KL_tr_loss += KL_loss.item()
            BOW_tr_loss += BOW_loss.item()
            NMI_tr += NMI
            n_correct_tr += n_correct

            # if iteration % 100 == 0:
            #     print("Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
            #               %(loss.data, NLL_loss.item()/args.batch_size, KL_loss.item()/args.batch_size, KL_weight))
            #     x_sentences = input[:3].cpu().numpy()
            #     print('\nInput sentences :')
            #     print(*idx2word(x_sentences, i2w=i2w, eos_idx=eos_idx), sep='\n')
            #     _, y_sentences = torch.topk(logp, 1, dim=-1)
            #     y_sentences = y_sentences[:3].squeeze().cpu().numpy()
            #     print('\nOutput sentences : ')
            #     print(*idx2word(y_sentences, i2w=i2w, eos_idx=eos_idx), sep='\n')
            #     print('\n')

        tr_loss     = tr_loss / len(datasets.train)
        NLL_tr_loss = NLL_tr_loss / len(datasets.train)
        KL_tr_loss  = KL_tr_loss / len(datasets.train)
        BOW_tr_loss  = BOW_tr_loss / len(datasets.train)
        NMI_tr = NMI_tr / len(datasets.train)
        acc_tr = n_correct_tr / len(datasets.train)
        
        # calculate the validation loss for this epoch
        val_loss = 0.0
        NLL_val_loss = 0.0
        KL_val_loss = 0.0
        BOW_val_loss = 0.0
        NMI_val = 0.0
        n_correct_val = 0.0
        acc_val = 0.0
        
        model.eval() # turn on evaluation mode
        for batch in tqdm(val_iter): 
            x, lengths = getattr(batch, args.input_type)
            target = x[:, 1:] # remove <sos>
            input = x[:, :-1] # remove <eos>
            lengths -= 1 # account for the removal
            input, target = to_device(input), to_device(target)
            if args.conditional != 'none':
                y = batch.intent.squeeze()
                y = to_device(y)
                sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
                y = y[sorted_idx]
            
            logp, mean, logv, logc, z, bow = model(input, lengths)
            # loss calculation
            NLL_loss, KL_losses, KL_weight, BOW_loss = loss_fn(logp, bow, target, lengths, mean, logv,
                                                   args.anneal_function, step, args.k1, args.x1, args.m1)
            
            KL_loss = torch.sum(KL_losses)
            loss = (NLL_loss + KL_weight * KL_loss) #/args.batch_size
            if args.bow_loss:
                loss += BOW_loss

            if args.conditional=='none':
                pred_labels = 0
                n_correct = 0
                NMI = 0
            else:
                if args.conditional=='supervised':
                    label_loss, label_weight = loss_labels(logc, y,
                                                       args.anneal_function, step, args.k2, args.x2, args.m2)
                    loss += label_weight * label_loss
                elif args.conditional=='unsupervised':
                    entropy = torch.sum(torch.exp(logc) * torch.log(model.n_classes * torch.exp(logc)))
                    loss += entropy                
                pred_labels = logc.data.max(1)[1].long()
                n_correct = pred_labels.eq(y.data).cpu().sum().float().item()
                NMI = normalized_mutual_info_score(y.cpu().detach().numpy(), torch.exp(logc).cpu().max(1)[1].numpy())
                
            val_loss += loss.item()
            NLL_val_loss += NLL_loss.item()
            KL_val_loss += KL_loss.item()
            BOW_val_loss += BOW_loss.item()
            NMI_val += NMI
            n_correct_val += n_correct
            
        val_loss     = val_loss / len(datasets.valid)
        NLL_val_loss = NLL_val_loss / len(datasets.valid)
        KL_val_loss  = KL_val_loss / len(datasets.valid)
        BOW_val_loss  = BOW_val_loss / len(datasets.valid)
        NMI_val = NMI_val / len(datasets.valid)
        acc_val = n_correct_val / len(datasets.valid)
        
        print('Epoch {} : train {:.6f} valid {:.6f}'.format(epoch, tr_loss, val_loss))
        print('Training   :  NLL loss : {:.6f}, KL loss : {:.6f}, BOW : {:.6f}, acc : {:.6f}'.format(NLL_tr_loss, KL_tr_loss, BOW_tr_loss, acc_tr))
        print('Validation :  NLL loss : {:.6f}, KL loss : {:.6f}, BOW : {:.6f}, acc : {:.6f}'.format(NLL_val_loss, KL_val_loss, BOW_val_loss, acc_val))

    run['NLL_hist'] = NLL_hist
    run['KL_hist'] = KL_hist
    run['NMI_hist'] = NMI_hist
    run['acc_hist'] = acc_hist
    run['latent'] = latent_rep

    return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='snips', choices=['snips', 'atis', 'sentiment', 'spam', 'yelp', 'penn-tree-bank'])
    parser.add_argument('--datasize', type=int, default=None)
    parser.add_argument('--model', type=str, default='CVAE')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pickle', type=str, default='run')
    parser.add_argument('-ng', '--n_generated', type=int, default=100)
    parser.add_argument('--benchmark', action='store_true')

    parser.add_argument('-it', '--input_type', type=str, default='delexicalised', choices=['delexicalised', 'utterance'])
    parser.add_argument('--conditional', type=str, default='supervised', choices=['supervised', 'unsupervised', 'none'])

    parser.add_argument('-msl', '--max_sequence_length', type=int, default=60)
    parser.add_argument('-mvs', '--max_vocab_size', type=int, default=10000)
    parser.add_argument('--emb_dim' , type=int, default=100)
    parser.add_argument('--emb_type' , type=str, default='glove', choices=['glove','none'])
    parser.add_argument('--tokenizer' , type=str, default='nltk', choices=['split', 'nltk', 'spacy'])
    parser.add_argument('--slot_averaging' , type=str, default='micro', choices=['none', 'micro', 'macro'])
    parser.add_argument('--bow_loss', type=bool, default=False)
    
    parser.add_argument('-ep', '--epochs', type=int, default=2)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru', choices=['rnn', 'gru', 'lstm'])
    parser.add_argument('-hs', '--hidden_size', type=int, default=128)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)

    parser.add_argument('-t', '--temperature', type=float, default=1)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic', choices=['logistic', 'linear'])
    parser.add_argument('-k1', '--k1', type=float, default=0.005, help='anneal time for KL weight')
    parser.add_argument('-x1', '--x1', type=int, default=500,     help='anneal rate for KL weight')
    parser.add_argument('-m1', '--m1', type=float, default=1.,    help='final value for KL weight')
    parser.add_argument('-k2', '--k2', type=float, default=0.01, help='anneal time for label weight')
    parser.add_argument('-x2', '--x2', type=int, default=100,      help='anneal rate for label weight')
    parser.add_argument('-m2', '--m2', type=float, default=1.,    help='final value for label weight')
    # parser.add_argument('-k3', '--k3', type=float, default=0.005, help='anneal time for word dropout')
    # parser.add_argument('-x3', '--x3', type=int, default=50,      help='anneal rate for word dropout')
    # parser.add_argument('-m3', '--m3', type=float, default=1.,    help='final value for word dropout')
    
    args = parser.parse_args()

    run = {} 
    run['args'] = args
    print(args)

    if args.dataset!='snips':
        args.input_type = 'utterance'
    args.pickle = args.pickle.rstrip('.pkl')

    datadir = os.path.join(args.dataroot, args.dataset)
    print('loading and embedding datasets')
    train_path = os.path.join(datadir, 'train.csv')
    validate_path = os.path.join(datadir, 'validate.csv')

    # Make a smaller dataset
    if args.datasize is not None:
        raw_path = train_path.replace('.csv', '_raw{}.csv'.format(args.datasize))
        train_csv = open(train_path, 'r')
        train_reader = list(csv.reader(train_csv))
        raw_csv = open(raw_path, 'w')
        raw_writer = csv.writer(raw_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        counter = 0
        import random
        while counter < args.datasize:
            row = random.choice(train_reader[1:]) # skip the header
            counter += 1
            raw_writer.writerow(row)
        train_csv.close()
        raw_csv.close()
        train_path = raw_path
        
    datasets = Datasets(train_path=os.path.join(train_path), valid_path=os.path.join(validate_path),
                        emb_dim=args.emb_dim, emb_type=args.emb_type,
                        max_vocab_size=args.max_vocab_size, max_sequence_length=args.max_sequence_length,
                        tokenizer=args.tokenizer)

    vocab = datasets.TEXT.vocab if args.input_type=='utterance' else datasets.DELEX.vocab
    i2w = vocab.itos
    w2i = vocab.stoi
    i2int = datasets.INTENT.vocab.itos
    int2i = datasets.INTENT.vocab.stoi
    n_classes = len(i2int)
    sos_idx = w2i['<sos>']
    eos_idx = w2i['<eos>']
    pad_idx = w2i['<pad>']
    unk_idx = w2i['<unk>']    

    if args.input_type=='delexicalised':
        print('embedding the slots with %s averaging' %args.slot_averaging)
        datasets.embed_slots(args.slot_averaging)
    print('embedding unknown words with random initialization')
    datasets.embed_unks(vocab, num_special_toks=4)
            
    model = getattr(models, args.model)(
            conditional=True if args.conditional!='none' else False,
            bow=args.bow_loss,
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
            n_classes=n_classes,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional,
            temperature=args.temperature
        )

    if args.load_model is not None:
        state_dict = torch.load(args.load_model)
        if state_dict['embedding.weight'].size(0) != model.embedding.weight.size(0): # vocab changed
            state_dict['embedding.weight'] = vocab.vectors
            state_dict['outputs2vocab.weight'] = torch.randn(len(i2w), args.hidden_size*model.hidden_factor)
            state_dict['outputs2vocab.bias'] = torch.randn(len(i2w))
        model.load_state_dict(state_dict)
    else:
        model.embedding.weight.data.copy_(vocab.vectors)
    
    model = to_device(model)
    print(model)

    NLL_recon = torch.nn.NLLLoss(reduction='sum', ignore_index=pad_idx)
    NLL_label = torch.nn.NLLLoss(reduction='sum')

    train(model, datasets, args)
    
    torch.save(model.state_dict(), args.pickle+'.pyT')
    
    if args.n_generated>0:

        generated = {}
        
        model.eval()

        samples, z, y_onehot, logp = model.inference(n=args.n_generated)
        samples = samples.cpu().numpy() 
        
        generated['samples'] = samples
        if args.conditional != 'none':
            intents = y_onehot.data.max(1)[1].cpu().numpy()
            generated['intents'] = [i2int[intent] for intent in intents]

        if args.input_type == 'delexicalised':
            delexicalised =  idx2word(samples, i2w=i2w, eos_idx=eos_idx)
            labellings, sentences = surface_realisation(samples, i2w=i2w, pad_idx=eos_idx)
            generated['delexicalised']=delexicalised
            generated['sentences']=sentences
        else:
            sentences = idx2word(samples, i2w=i2w, eos_idx=eos_idx)
            generated['sentences']=sentences

        print('----------GENERATED----------')
        for i in range(args.n_generated):
            if args.conditional != 'none':
                print('Intents   : ', i2int[intents[i]])
            if args.input_type == 'delexicalised':
                print('Delexicalised : ', delexicalised[i])
            print('Sentences : ', sentences[i]+'\n')

        bleu_scores = calc_bleu(sentences, intents, datasets)
        diversity = calc_diversity(sentences, datasets)
        entropy = calc_entropy(logp)
        intent_accuracy = intent_classification(samples, intents)
        print('BLEU quality : ', bleu_scores['quality'])
        print('BLEU diversity : ', bleu_scores['diversity'])
        print('Diversity : ', diversity)
        print('Entropy : ', entropy)        
        run['generated'] = generated
        run['bleu_scores'] = bleu_scores
        run['diversity'] = diversity
        run['entropy'] = entropy

        if args.benchmark:
            from snips_nlu import SnipsNLUEngine
            from snips_nlu_metrics import compute_train_test_metrics

            augmented_path = train_path.replace('.csv', '_aug{}.csv'.format(args.datasize, args.n_generated))
            print('Dumping augmented dataset at %s' %augmented_path)
            from shutil import copyfile
            
            copyfile(train_path, augmented_path)
            augmented_csv = open(augmented_path, 'a')
            augmented_writer = csv.writer(augmented_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for s, l, d, i in zip(sentences, labellings, delexicalised, intents):
                augmented_writer.writerow([s, l, d, i2int[i]])
            
            csv2json(train_path)
            csv2json(augmented_path)

            print('Starting benchmarking...')

            def my_matching_lambda(lhs_slot, rhs_slot):
                return lhs_slot['text'].strip() == rhs_slot["rawValue"].strip()

            raw_metrics = compute_train_test_metrics(train_dataset=os.path.join(datadir, 'train.json'),
                                                    test_dataset=os.path.join(datadir, 'validate.json'),
                                                    engine_class=SnipsNLUEngine,
                                                    slot_matching_lambda = my_matching_lambda
                                                    )
            augmented_metrics = compute_train_test_metrics(train_dataset=os.path.join(datadir, 'train_augmented.json'),
                                                    test_dataset=os.path.join(datadir, 'validate.json'),
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
    
    torch.save(run, args.pickle+'.pkl')
