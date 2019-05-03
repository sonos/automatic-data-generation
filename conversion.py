import json
import argparse
import os 
import csv
import pickle

def json2csv(datadir, outdir):
    
    print('starting conversion')
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
            for sentence in data:
                utterance = ''
                labelling = ''
                delexicalised = ''
                delexicalised = 'SOS '

                for group in sentence['data']:
                    words = group['text']
                    if args.remove_punctuation :
                        for p in punctuation:
                            words = words.replace(p, '')
                    utterance += words
                    
                    if 'entity' in group.keys(): #this group is a slot
                        slot = group['entity'].lower()
                        if args.remove_punctuation :
                            for p in punctuation:
                                slot = slot.replace(p, '')

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
                  
                delexicalised += ' EOS'
                        
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


def csv2json(datadir, outdir):
    
    print('starting conversion')
    punctuation = [',', '.', ':', ';', '?', '!']

    json_files = {}

    for intent in intents:
        json_files[intent] = open(os.path.join(outdir,intent,'{}_{}{}.json'.format(split, intent, '_full' if split=='train' else '')), encoding='ISO-8859-1')
        dic[intent] = {'root': {intent: {} } }

    for split in ['train','validate']:
        
        csvfile    = open('{}/{}.csv'               .format(outdir, split), 'r')
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')

        for i, irow in enumerate(spamreader):
            
            utterance, labelling, delexicalised, intent = *row

            dic[intent]['root'][intent][irow] = {'data': {} }
            
            group_idx = 0
            for (word, label) in zip(utterance.split(), labelling.split()):
                new_inside_slot = label.split('-'[1:])
                if new_inside_slot != old_inside_slot:
                    if old_inside_slot : #the group is a slot
                    dic[intent]['root'][intent][irow]['data'][group_idx] = {'text' : text, 'entity':}
                    group_idx +=1
                else :
                    text += word
            
#             dic[intent]['root'][intent][row][]

#             for intent, data in datadic.items():
#                 for sentence in data:
#                 utterance = ''
#                 labelling = ''
#                 delexicalised = ''
#                 delexicalised = 'SOS '

#                 for group in sentence['data']:
#                     words = group['text']
#                     if args.remove_punctuation :
#                         for p in punctuation:
#                             words = words.replace(p, '')
#                             utterance += words
                            
#                     if 'entity' in group.keys(): #this group is a slot
#                         slot = group['entity'].lower()
#                         if args.remove_punctuation :
#                             for p in punctuation:
#                                 slot = slot.replace(p, '')

#                         delexicalised += '_'+slot+'_'
#                         for i, word in enumerate(words.split(' ')):
#                             if word == '':
#                                 continue
#                             if i==0:
#                                 word = 'B-'+slot+' '
#                             else:
#                                 word = 'I-'+slot+' '
#                                 labelling += word
                                
#                         if slot not in slotdic.keys():
#                             slotdic[slot] = [words]
#                         else:
#                             if words not in slotdic[slot]:
#                                 slotdic[slot].append(words)
                                
#                     else : #this group is just context
#                         delexicalised += words
#                         labelling += '0 '
                        
#                 delexicalised += ' EOS'
                
#                 intentfile.write(intent+'\n')
#                 utterfile.write(utterance+'\n')
#                 labelfile.write(labelling+'\n')
#                 delexfile.write(delexicalised+'\n')
#                 csv_writer.writerow([utterance, labelling, delexicalised, intent])
                
#         with open('{}/{}_slot_values'.format(outdir, split), 'wb') as f:
#             pickle.dump(slotdic, f)
#             print('dumped dic')
            
#     print('Example : ')
#     print('Original utterance : ', utterance)
#     print('Labelled : ', labelling)   
#     print('Delexicalised : ', delexicalised)   
#     print('done')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='/Users/stephane/Dropbox/Work/Codes/data/2017-06-custom-intent-engines')
    parser.add_argument('--outdir' , type=str, default='./data/')
    parser.add_argument('--remove_punctuation' , type=int, default=1)
    args = parser.parse_args()
    
    if not os.path.isdir(args.outdir):
        print('saving data to', args.outdir)
        os.mkdir(args.outdir)
    
    json2csv(args.datadir, args.outdir)

    
