import json
import argparse
import os 

def convert(datadir):
    
    print('starting conversion')

    datadic = {}
    for intent in os.listdir(datadir):
        with open(os.path.join(datadir,intent,'train_{}_full.json'.format(intent)), encoding='ISO-8859-1') as json_file:
            datadic[intent] = json.load(json_file)[intent]

    intentfile = open('intents', 'w')
    utterfile  = open('utterances', 'w')
    labelfile  = open('labels', 'w')
    for intent, data in datadic.items():
        for sentence in data:
            utterance = ''
            labelling = ''
            for group in sentence['data']:
                words = group['text']
                try:
                    label = group['entity']
                except:
                    label = None
                utterance += words
                separated_words = words.split(' ')
                for i, word in enumerate(separated_words):
                    if word=='':
                        continue
                    if label is None: #no slot
                        labelling += '0 '
                    else: #start slot
                        if i==0:
                            word = 'B-'+label+' '
                        else:
                            word = 'I-'+label+' '
                        labelling += word

            intentfile.write(intent+'\n')
            utterfile.write(utterance+'\n')
            labelfile.write(labelling+'\n')

    print('done')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='/Users/stephane/Dropbox/Work/Codes/data/2017-06-custom-intent-engines')
    args = parser.parse_args()
    
    convert(args.datadir)

    