import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from embedding import Datasets
from tqdm import tqdm
import argparse

def intent_classification(emb_dim=100, rnn_type='GRU'):

    datasets = Datasets(train_path='./data/train.csv', valid_path='./data/validate.csv', emb_dim=100)
    train_iter, val_iter = datasets.get_iterators(batch_size=64)

    TEXT = datasets.train.fields['utterance']

    class RNN_classifier(nn.Module):
        def __init__(self, hidden_dim, emb_dim=300, num_linear=3, n_classes=8, bidirectional=False, rnn_type='GRU'):
            super().__init__() # don't forget to call this!
            self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
            if rnn_type == 'GRU':
                self.encoder = nn.GRU(emb_dim, hidden_dim, num_layers=1)
            elif rnn_type == 'LSTM':
                self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1)
            elif rnn_type == 'RNN':
                self.encoder = nn.RNN(emb_dim, hidden_dim, num_layers=1)
            self.linear_layers = []
            for i in range(num_linear - 1):
                self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers = nn.ModuleList(self.linear_layers)
            self.predictor = nn.Linear(hidden_dim, n_classes)

        def forward(self, seq):
            hdn, _ = self.encoder(self.embedding(seq))
            feature = hdn[-1, :, :]
            for layer in self.linear_layers:
                feature = layer(feature)
            preds = self.predictor(feature)
            return preds

    nh = 500
    nl = 3
    model = RNN_classifier(nh, emb_dim=emb_dim, n_classes=7, bidirectional=True, rnn_type=rnn_type)

    opt = optim.Adam(model.parameters(), lr=1e-2)
    loss_func = F.nll_loss

    epochs = 2

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_corrects = 0
        model.train() # turn on training mode
        for batch in tqdm(train_iter): 
            x = batch.utterance        #x = batch.utterance
            y = batch.intent
            opt.zero_grad()
            preds = model(x)
            preds = F.log_softmax(preds, dim=1)
            y = y-1
            loss = loss_func(preds, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            pred_labels = preds.data.max(1)[1].long()
            running_corrects += pred_labels.eq(y.data).cpu().sum()

        epoch_loss     = running_loss / len(datasets.train)
        epoch_corrects = running_corrects.float() / len(datasets.train)

        # calculate the validation loss for this epoch
        val_loss = 0.0
        val_corrects = 0
        model.eval() # turn on evaluation mode
        for batch in tqdm(val_iter): 
            x = batch.utterance        #x = batch.utterance
            y = batch.intent        
            preds = model(x)
            preds = F.log_softmax(preds, dim=1)
            y = y-1
            loss = loss_func(preds, y)

            val_loss += loss.item() * x.size(0)
            pred_labels = preds.data.max(1)[1].long()
            val_corrects += pred_labels.eq(y.data).cpu().sum()

        val_loss     = val_loss / len(datasets.valid)
        val_corrects = val_corrects.float() / len(datasets.valid)
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
        print('Epoch: {}, Training Accuracy: {:.4f}, Validation Accuracy: {:.4f}'.format(epoch, epoch_corrects, val_corrects))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_type', type=str, default='GRU')
    parser.add_argument('--emb_dim' , type=int, default=100)
    args = parser.parse_args()
    
    intent_classification(emb_dim=args.emb_dim, rnn_type=args.rnn_type)
