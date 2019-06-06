import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from automatic_data_generation.utils.utils import to_device
import ipdb

class CVAE(nn.Module):

    def __init__(self, conditional, bow, vocab_size, embedding_size, rnn_type, hidden_size,
                 word_dropout=0, embedding_dropout=0, z_size=100, n_classes=10,
                 sos_idx=0, eos_idx=0, pad_idx=0, unk_idx=0,
                 max_sequence_length=30, num_layers=1, bidirectional=False,
                 temperature=1):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() \
            else torch.Tensor

        self.conditional = conditional
        self.bow = bow
        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.z_size = z_size
        self.n_classes = n_classes
        self.latent_size = z_size + n_classes if conditional else z_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.temperature = temperature

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            batch_first=True)

        self.decoder_rnn = rnn(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True
        )  # decoder should be unidirectional

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, z_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, z_size)

        if conditional : self.hidden2cat = nn.Linear(hidden_size * self.hidden_factor,
                                    n_classes)
        self.latent2hidden = nn.Linear(self.latent_size,
                                       hidden_size * self.num_layers)

        if self.bow:
            self.z2bow = nn.Sequential(
                nn.Linear(self.z_size,
                          int((self.z_size + vocab_size) / 2)),
                nn.Tanh(),
                nn.Linear(int((self.z_size + vocab_size) / 2), vocab_size)
            )
        self.outputs2vocab = nn.Linear(hidden_size,
                                       vocab_size) 

    def forward(self, input_sequence, lengths):

        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        input_sequence = input_sequence[sorted_idx]
        
        # ENCODER
        input_embedding = self.embedding(input_sequence)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)
        z = to_device(torch.randn(batch_size, self.z_size))
        z = z * std + mean

        if self.conditional:
            logc = nn.functional.log_softmax(self.hidden2cat(hidden), dim=-1)
            y_onehot = nn.functional.gumbel_softmax(logc)
            latent = torch.cat((z, y_onehot), dim=-1)
        else:
            logc = None
            latent = z

        # DECODER
        hidden = self.latent2hidden(latent)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.num_layers, batch_size, self.hidden_size)
            # hidden = hidden.view(self.hidden_factor, batch_size,
            # self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)
            
        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            prob = to_device(prob)
            prob[(input_sequence.data - self.sos_idx) * (
                    input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[
                prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        input_embedding = self.embedding_dropout(input_embedding)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        outputs, _ = self.decoder_rnn(packed_input, hidden)


        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        bs, seqlen, hs = padded_outputs.size()

        logits = self.outputs2vocab(padded_outputs.view(-1, hs))
        logp = nn.functional.log_softmax(logits / self.temperature, dim=-1)
        logp = logp.view(bs, seqlen, self.embedding.num_embeddings)

        if self.bow:
            bow = nn.functional.log_softmax(self.z2bow(z), dim=0)
            bow = bow[reversed_idx]
        else:
            bow=None
            
        return logp, mean, logv, logc, z, bow

    def inference(self, n=10, z=None, y_onehot=None, temperature=0):

        if z is None:
            batch_size = n
            z = torch.randn(batch_size, self.z_size)
        else:
            batch_size = z.size(0)

        if self.conditional:
            if y_onehot is None:
                y = torch.LongTensor(batch_size, 1).random_() % self.n_classes
                y_onehot = torch.FloatTensor(batch_size, self.n_classes)
                y_onehot.fill_(0.001)
                y_onehot.scatter_(dim=1, index=y, value=1)
            latent = to_device(torch.cat((z, y_onehot), dim=1))                
        else:
            y_onehot = None
            latent = to_device(z)
            
        hidden = self.latent2hidden(latent)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.num_layers, batch_size, self.hidden_size)
            # hidden = hidden.view(self.hidden_factor, batch_size,
            # self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(
            0, batch_size, out=self.tensor()).long()  # all idx of batch
        sequence_running = torch.arange(
            0, batch_size, out=self.tensor()).long()  # all idx of batch
        # which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(
            0, batch_size, out=self.tensor()).long()  # idx of still
        # generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(
            self.pad_idx).long()

        t = 0
        while t < self.max_sequence_length and len(running_seqs) > 0:
            if t == 0:
                input_sequence = torch.Tensor(batch_size).fill_(
                    self.sos_idx).long()
                # input_sequence = torch.randint(0, self.vocab_size,
                # (batch_size,))

            input_sequence = to_device(input_sequence.unsqueeze(1))

            input_embedding = self.embedding(input_sequence)
            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)
            if t == 0: # prevent from generating empty sentence
                logits[:,:,self.eos_idx] = torch.min(logits, dim=-1)[0]
            logp = nn.functional.log_softmax(logits / self.temperature, dim=-1)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence,
                                            sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (
                    input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                try:
                    input_sequence = input_sequence[running_seqs]
                except:
                    break
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs),
                                            out=self.tensor()).long()

            t += 1

        return generations, z, y_onehot, logp

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:, t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to

##########################################################################################
    
class VAE(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size,
                 word_dropout=0, embedding_dropout=0, z_size=100, n_classes=10,
                 sos_idx=0, eos_idx=0, pad_idx=0, unk_idx=0,
                 max_sequence_length=30, num_layers=1, bidirectional=False,
                 temperature=1, conditional=False, bow=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.conditional = conditional
        self.bow = bow
        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.z_size = z_size
        self.n_classes = n_classes
        self.latent_size = z_size + n_classes if conditional else z_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.temperature = temperature

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, self.latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, self.latent_size)
        self.latent2hidden = nn.Linear(self.latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def forward(self, input_sequence, length):

        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_device(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean

        # DECODER
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(batch_size, self.hidden_factor, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b, s, _ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)


        return logp, mean, logv, None, z, None


    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            z = to_device(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.num_layers, batch_size, self.hidden_size)
            # hidden = hidden.view(self.hidden_factor, batch_size,
            # self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t=0
        while(t<self.max_sequence_length and len(running_seqs)>0):

            if t == 0:
                input_sequence = to_device(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z, None

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to

