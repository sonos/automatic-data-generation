import numpy as np
from automatic_data_generation.models.embedding import Datasets
from automatic_data_generation.models.cvae import CVAE
from tqdm import tqdm
import argparse
import os
import torch
from automatic_data_generation.utils.utils import to_device, idx2word, surface_realisation
from sklearn.metrics import normalized_mutual_info_score
import csv
from automatic_data_generation.utils.conversion import csv2json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import copy


def anneal_fn(anneal_function, step, k, x, m):
    if anneal_function == 'logistic':
        return m*float(1/(1+np.exp(-k*(step-x))))
    elif anneal_function == 'linear':
        return m*min(1, step/x)


def loss_fn(logp, bow, target, mean, logv, anneal_function, step, k1, x1, m1):

    batch_size = target.size(1)

    # Bag of words
    bow.view(batch_size,-1)
    target = target.view(batch_size,-1)
    BOW_loss = - torch.einsum('iik->', bow[:,target])

    target = target.view(-1)
    logp = logp.view(-1, logp.size(2))
    
    # Negative Log Likelihood
    NLL_loss = NLL(logp, target)

    # KL Divergence
    KL_losses = -0.5 * torch.sum((1 + logv - mean.pow(2) - logv.exp()), dim=0)
    KL_weight = anneal_fn(anneal_function, step, k1, x1, m1)
    
    return NLL_loss, KL_losses, KL_weight, BOW_loss


def loss_labels(logc, target, anneal_function, step, k2, x2, m2):
    
    # Negative Log Likelihood
    label_loss = NLL(logc, target)
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
        for batch in tqdm(train_iter): 
            step += 1
            opt.zero_grad()
            # model.word_dropout_rate =  anneal_fn(args.anneal_function, step, args.k3, args.x3, args.m3)

            x = getattr(batch, args.input_type)
            y = batch.intent.squeeze() #torch.ones_like(batch.intent) #
            x, y = to_device(x), to_device(y)
            
            logp, mean, logv, logc, z, bow = model(x)
            if epoch == args.epochs:
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
            NLL_loss, KL_losses, KL_weight, BOW_loss = loss_fn(logp, bow, x, mean, logv,
                                                   args.anneal_function, step, args.k1, args.x1, args.m1)
            KL_loss = torch.sum(KL_losses)
            NLL_hist.append(NLL_loss.detach().cpu().numpy()/args.batch_size)
            KL_hist.append(KL_losses.detach().cpu().numpy()/args.batch_size)
            BOW_hist.append(BOW_loss.detach().cpu().numpy()/args.batch_size)
            loss = (NLL_loss + KL_weight * KL_loss) #/args.batch_size
            if args.bow_loss:
                loss+BOW_loss

            if args.supervised:
                label_loss, label_weight = loss_labels(logc, y,
                                                       args.anneal_function, step, args.k2, args.x2, args.m2)
                loss += label_weight * label_loss
            else:
                entropy = torch.sum(c * torch.log(model.n_classes * c))
                loss += entropy

            pred_labels = logc.data.max(1)[1].long()
            n_correct = pred_labels.eq(y.data).cpu().sum().float().item()
            acc_hist.append(n_correct/args.batch_size)
            NMI = normalized_mutual_info_score(y.cpu().detach().numpy(), c.cpu().max(1)[1].numpy())
            NMI_hist.append(NMI)                
                
            loss.backward()
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
            x = getattr(batch, args.input_type)
            y = batch.intent
            x, y = to_device(x), to_device(y) 
            
            logp, mean, logv, logc, z, bow = model(x)
            c = torch.exp(logc)
            
            # loss calculation
            NLL_loss, KL_losses, KL_weight, BOW_loss = loss_fn(logp, bow, x, mean, logv,
                                                   args.anneal_function, step, args.k1, args.x1, args.m1)
            
            KL_loss = torch.sum(KL_losses)
            loss = (NLL_loss + KL_weight * KL_loss) #/args.batch_size
            if args.bow_loss:
                loss+BOW_loss

            if args.supervised:
                label_loss, label_weight = loss_labels(logc, y,
                                                       args.anneal_function, step, args.k2, args.x2, args.m2)
                loss += label_weight * label_loss
            else:
                entropy = torch.sum(c * torch.log(model.n_classes * c))
                #loss += entropy

            pred_labels = logc.data.max(1)[1].long()             
            n_correct = pred_labels.eq(y.data).cpu().sum().float().item()
            NMI = normalized_mutual_info_score(y.cpu().detach().numpy(), c.cpu().max(1)[1].numpy())

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
    run['mean'] = mean
    run['logv'] = logv

    return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='snips', choices=['snips', 'atis', 'sentiment', 'spam', 'yelp'])
    # parser.add_argument('--train_path', type=str, default='./data/snips/train.csv')
    # parser.add_argument('--validate_path', type=str, default='./data/snips/validate.csv')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--save_model', type=str, default='model.pyT')
    parser.add_argument('--pickle', type=str, default='run.pyT')
    parser.add_argument('-spi', '--samples_per_intent', type=int, default=1000)
    parser.add_argument('-ng', '--n_generated', type=int, default=100)
    parser.add_argument('--benchmark', action='store_true')

    parser.add_argument('-it', '--input_type', type=str, default='delexicalised', choices=['delexicalised', 'utterance'])
    parser.add_argument('--supervised', type=bool, default=True)
    parser.add_argument('-pr', '--print_reconstruction', type=int, default=-1, help='Print the reconstruction at a given epoch')

    parser.add_argument('-msl', '--max_sequence_length', type=int, default=8)
    parser.add_argument('--emb_dim' , type=int, default=100)
    parser.add_argument('--tokenizer' , type=str, default='nltk', choices=['split', 'nltk', 'spacy'])
    parser.add_argument('--slot_averaging' , type=str, default='micro', choices=['none', 'micro', 'macro'])
    parser.add_argument('--bow_loss', type=bool, default=False)
    
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
    parser.add_argument('-k1', '--k1', type=float, default=0.005, help='anneal time for KL weight')
    parser.add_argument('-x1', '--x1', type=int, default=100,     help='anneal rate for KL weight')
    parser.add_argument('-m1', '--m1', type=float, default=1.,    help='final value for KL weight')
    parser.add_argument('-k2', '--k2', type=float, default=0.005, help='anneal time for label weight')
    parser.add_argument('-x2', '--x2', type=int, default=50,      help='anneal rate for label weight')
    parser.add_argument('-m2', '--m2', type=float, default=1.,    help='final value for label weight')
    # parser.add_argument('-k3', '--k3', type=float, default=0.005, help='anneal time for word dropout')
    # parser.add_argument('-x3', '--x3', type=int, default=50,      help='anneal rate for word dropout')
    # parser.add_argument('-m3', '--m3', type=float, default=1.,    help='final value for word dropout')

    run = {}

    args = parser.parse_args()
    run['args'] = args
    print(args)
    
    # datadir = os.path.dirname(args.train_path)
    #json2csv(datadir+'/2017-06-custom-intent-engines', datadir, samples_per_intent=args.samples_per_intent)

    print('loading and embedding datasets')
    datadir = os.path.join('./data', args.dataset)
    train_path = os.path.join(datadir, 'train.csv')
    validate_path = os.path.join(datadir, 'validate.csv')
    datasets = Datasets(train_path=os.path.join(train_path), valid_path=os.path.join(validate_path), emb_dim=args.emb_dim, tokenizer=args.tokenizer)
    
    vocab = datasets.TEXT.vocab if args.input_type=='utterance' else datasets.DELEX.vocab
    i2w = vocab.itos
    w2i = vocab.stoi
    i2int = datasets.INTENT.vocab.itos
    int2i = datasets.INTENT.vocab.stoi
    n_classes = len(i2int)
    sos_idx = w2i['SOS']
    eos_idx = w2i['EOS']
    pad_idx = w2i['<pad>']
    unk_idx = w2i['<unk>']
    
    # if args.input_type=='delexicalised':
    #     print('embedding the slots with %s averaging' %args.slot_averaging)
    #     datasets.embed_slots(args.slot_averaging)
    print('embedding unknown words with random initialization')
    datasets.embed_unks(vocab, num_special_toks=4)
    
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
    
    train(model, datasets, args)
    if args.save_model is not None:
        torch.save(model.state_dict(), args.save_model)
    
    if args.n_generated>0:
    
        model.eval()

        samples, z, y_onehot = model.inference(n=args.n_generated)
        intents = y_onehot.data.max(1)[1].cpu().numpy()
        sentences = idx2word(samples, i2w=i2w, pad_idx=pad_idx)

        generated = {'samples':samples, 'intents':[i2int[intent] for intent in intents], 'sentences':sentences}
        if args.input_type == 'delexicalised':
            delexicalised = copy.deepcopy(sentences)
            labellings, sentences = surface_realisation(samples, i2w=i2w, pad_idx=pad_idx)
            generated['delexicalised']=delexicalised

        print('----------GENERATED----------')
        for i in range(args.n_generated):
            print('Intents   : ', i2int[intents[i]])
            if args.input_type == 'delexicalised':
                print('Delexicalised : ', delexicalised[i])
            print('Sentences : ', sentences[i]+'\n')

        bleu_scores = {}
        cc =SmoothingFunction()
        references = {intent:[] for intent in range(model.n_classes)}
        candidates = {intent:[] for intent in range(model.n_classes)}
        for example in datasets.train:
            references[int2i[example.intent]].append(example.utterance)
        for i, example in enumerate(sentences):
            candidates[intents[i]].append(datasets.tokenize(example))
        for intent in range(model.n_classes):
            bleu_scores[i2int[intent]] = np.mean([sentence_bleu(references[intent], candidate, weights=[1, 0, 0, 0], smoothing_function=cc.method1) for candidate in candidates[intent]])
        avg_bleu_score = np.mean([bleu_score for bleu_score in bleu_scores.values()])
        print('BLEU scores : ', bleu_scores)
        print('Average BLEU : ', avg_bleu_score)
        bleu_scores['average'] = avg_bleu_score

        tokens = np.concatenate([datasets.tokenize(sentence) for sentence in sentences])
        diversity = len(set(tokens))/float(len(tokens))
        print('Diversity : ', diversity)
        
        run['generated'] = generated
        run['bleu_scores'] = bleu_scores
        run['diversity'] = diversity

        if args.benchmark:
            from snips_nlu import SnipsNLUEngine
            from snips_nlu_metrics import compute_train_test_metrics

            augmented_path = train_path.replace('.csv', '_augmented.csv')
            print('Dumping augmented dataset at %s' %augmented_path)
            from shutil import copyfile
            copyfile(train_path, augmented_path)
            csvfile    = open(augmented_path, 'a')
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for s, l, d, i in zip(sentences, labellings, delexicalised, intents):
                csv_writer.writerow([s, l, d, i2int[i]])
            
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
