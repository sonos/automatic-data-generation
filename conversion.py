import json
import argparse
import os 
import csv
import pickle
from nltk import word_tokenize

remove_punctuation = True

def json2csv(datadir, outdir, samples_per_class):
    
    print('Starting json2csv conversion...')
    punctuation = [',', '.', ':', ';', '?', '!']
    
    for split in ['train','validate']:
    
        datadic = {}
        for intent in os.listdir(datadir):
            with open(os.path.join(datadir,intent,'{}_{}{}.json'.format(split, intent, '_full' if split=='train' else '')), encoding='ISO-8859-1') as json_file:
                datadic[intent] = json.load(json_file)[intent]
                
        slotdic = {}

        intentfile = open('{}/{}_intents.txt'       .format(outdir, split), 'w')
        utterfile  = open('{}/{}_utterances.txt'    .format(outdir, split), 'w')
        labelfile  = open('{}/{}_labels.txt'        .format(outdir, split), 'w')
        delexfile  = open('{}/{}_delexicalised.txt' .format(outdir, split), 'w')
        csvfile    = open('{}/{}.csv'               .format(outdir, split), 'w')
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['utterance', 'labels', 'delexicalised', 'intent'])

        for intent, data in datadic.items():
            for isent, sentence in enumerate(data):
                if isent >= samples_per_class:
                    break
                utterance = ''
                labelling = ''
                #delexicalised = ''
                delexicalised = 'SOS '

                for group in sentence['data']:
                    words = group['text']
                    if remove_punctuation :
                        for p in punctuation:
                            words = words.replace(p, ' ')
                    utterance += words
                    
                    if 'entity' in group.keys(): #this group is a slot
                        slot = group['entity'].lower()
                        if remove_punctuation :
                            for p in punctuation:
                                slot = slot.replace(p, '')

                        delexicalised += '_'+slot+'_'
                        for i, word in enumerate(word_tokenize(words)):
                            # if word == '':
                            #     continue
                            if i==0:
                                word = 'B-'+slot+' '
                            else:
                                word = 'I-'+slot+' '
                            labelling += word
                            
                        if slot not in slotdic.keys():
                            slotdic[slot] = [words]
                        else:
                            if words not in slotdic[slot]:
                                slotdic[slot].append(words)
                    
                    else : #this group is just context
                        delexicalised += words
                        labelling += 'O '*len(word_tokenize(words))
                  
                delexicalised += ' EOS'
                        
                intentfile.write(intent+'\n')
                utterfile.write(utterance+'\n')
                labelfile.write(labelling+'\n')
                delexfile.write(delexicalised+'\n')
                csv_writer.writerow([utterance, labelling, delexicalised, intent])
                
        with open('{}/{}_slot_values'.format(outdir, split), 'wb') as f:
            print(slotdic.keys())
            pickle.dump(slotdic, f)
            print('Dumped slot dictionnary')
            
    print('Example : ')
    print('Original utterance : ', utterance)
    print('Labelled : ', labelling)   
    print('Delexicalised : ', delexicalised)   

    print('Successfully converted csv2json !')

def get_groups(zipped):

    prev_label = None
    groups = []

    for i,(word, label) in enumerate(zipped):
        if label.startswith('B-'): #start slot group
            if i!=0 :
                groups.append(group) #dump previous group
            slot = label.lstrip('B-')
            group = {'text': (word+' '), 'entity':slot, 'slot_name':slot}
        elif (label=='O' and prev_label!='O'): #start context group
            if i!=0 :
                groups.append(group) #dump previous group
            group = {'text': (word+' ')}
        else:
            group['text'] += (word+' ')
        prev_label = label
    groups.append(group)

    return groups

def csv2json(datadir, outdir, augmented):

    print('Starting csv2json conversion...')

    jsondic = {'language':'en'}
    intents = {}
    entities = {}

    for split in ['train' if not augmented else 'train_augmented','validate']:

        encountered_slot_values = {}

        csvname = '{}/{}.csv'.format(datadir, split)
        csvfile = open(csvname, 'r')
        reader = csv.reader(csvfile)

        for irow, row in enumerate(reader):

            if irow==0: #ignore header
                continue

            utterance, labelling, delexicalised, intent = row
            if intent not in intents.keys():
                intents[intent] = {'utterances':[]}
            zipped = zip(word_tokenize(utterance), word_tokenize(labelling))
            groups = get_groups(zipped)
            intents[intent]['utterances'].append({'data':groups})
            for group in groups:
                if 'slot_name' in group.keys():
                    slot_name = group['slot_name']
                    slot_value = group['text']
                    if slot_name not in encountered_slot_values.keys():
                        encountered_slot_values[slot_name] = []

                    if slot_name not in entities.keys():
                        entities[slot_name] = {"data": [],
                                               "use_synonyms": True,
                                               "automatically_extensible": True,
                                               "matching_strictness": 1.0
                                               }
                    if slot_value not in encountered_slot_values[slot_name]:
                        entities[slot_name]['data'].append({'value': slot_value,
                                                            'synonyms':[]})
                    if slot_value == 'added ':
                        print(groups, utterance, labelling)

                    encountered_slot_values[slot_name].append(slot_value)

        jsondic['intents'] = intents
        jsondic['entities'] = entities

        jsonname = '{}/{}.json'.format(outdir, split)
        with open(jsonname, 'w') as jsonfile:
            json.dump(jsondic, jsonfile)

    print('Successfully converted csv2json !')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='/Users/stephane/Dropbox/Work/Codes/data/2017-06-custom-intent-engines')
    parser.add_argument('--outdir' , type=str, default='./data/')
    parser.add_argument('--samples_per_class' , type=int, default=100)
    parser.add_argument('--augmented' , type=int, default=1)
    parser.add_argument('--convert_to' , type=str, default='csv')
    parser.add_argument('--remove_punctuation' , type=int, default=1)
    args = parser.parse_args()
    
    if not os.path.isdir(args.outdir):
        print('saving data to', args.outdir)
        os.mkdir(args.outdir)

    if args.convert_to == 'csv':
        json2csv(args.datadir, args.outdir, args.samples_per_class)
    if args.convert_to == 'json':
        csv2json(args.datadir, args.outdir, args.augmented)
    
