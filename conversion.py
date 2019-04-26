import json
import argparse
import os 
import csv
import pickle

def convert(datadir, outdir):
    
    print('starting conversion')
    
    for split in ['train','validate']:
    
        datadic = {}
        for intent in os.listdir(datadir):
            with open(os.path.join(datadir,intent,'{}_{}{}.json'.format(split, intent, '_full' if split=='train' else '')), encoding='ISO-8859-1') as json_file:
                datadic[intent] = json.load(json_file)[intent]
                
        slotdic = {}

        intentfile = open('{}/{}_intents'       .format(outdir, split), 'w')
        utterfile  = open('{}/{}_utterances'    .format(outdir, split), 'w')
        labelfile  = open('{}/{}_labels'        .format(outdir, split), 'w')
        delexfile  = open('{}/{}_delexicalised' .format(outdir, split), 'w')
        csvfile    = open('{}/{}.csv'           .format(outdir, split), 'w')
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['utterance', 'labels', 'delexicalised', 'intent'])

        for intent, data in datadic.items():
            for sentence in data:
                utterance = ''
                labelling = ''
                delexicalised = ''
                #delexicalised = '# '

                for group in sentence['data']:
                    
                    words = group['text']
                    utterance += words
                    
                    if 'entity' in group.keys(): #this group is a slot
                        slot = group['entity'].lower()
                        delexicalised += '_'+slot+'_'
                        for i, word in enumerate(words.split(' ')):
                            if word == '':
                                continue
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
                        labelling += '0 '
                  
                if delexicalised[-1] != '.':
                    delexicalised += '.'
                #delexicalised += '<eos>'
                        
                intentfile.write(intent+'\n')
                utterfile.write(utterance+'\n')
                labelfile.write(labelling+'\n')
                delexfile.write(delexicalised+'\n')
                csv_writer.writerow([utterance, labelling, delexicalised, intent])
                
        with open('{}/{}_slot_values'.format(outdir, split), 'wb') as f:
            pickle.dump(slotdic, f)
            print('dumped dic')
            
    print('Example : ')
    print('Original utterance : ', utterance)
    print('Labelled : ', labelling)   
    print('Delexicalised : ', delexicalised)   
    print('done')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='/Users/stephane/Dropbox/Work/Codes/data/2017-06-custom-intent-engines')
    parser.add_argument('--outdir' , type=str, default='./data/')
    args = parser.parse_args()
    
    if not os.path.isdir(args.outdir):
        print('saving data to', args.outdir)
        os.mkdir(args.outdir)
    
    convert(args.datadir, args.outdir)

    
