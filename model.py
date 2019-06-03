from config import Config

import torch
import torch.nn as nn


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, taregt_size, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.target_size = taregt_size

        self.char_embedds = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2,
                            num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)

        self.hidden2tag = nn.Linear(hidden_dim, self.target_size)

        self.transitions = nn.Parameter(
            torch.randn(self.target_size, self.target_size))
        self.transitions.data[Config.START, :] = -10000.
        self.transitions.data[:, Config.STOP] = -10000.

    @staticmethod
    def log_sum_exp(input, keepdim=False):
        assert input.dim() == 2
        max_scores, _ = input.max(dim=-1, keepdim=keepdim)
        output = input - input.max(dim=-1, keepdim=True)[0]
        return max_scores + torch.log(torch.sum(torch.exp(output), dim=-1, keepdim=keepdim))

    @staticmethod
    def gather_index(input, index):
        assert input.dim() == 2 and index.dim() == 1
        index = index.unsqueeze(1).expand_as(input)
        output = torch.gather(input, 1, index)
        return output[:, 0]

    def _forward_alg(self, feats):
        bsz, sent_len, l_size = feats.size()
        if Config.use_gpu:
            init_alphas = torch.cuda.FloatTensor(
                bsz, self.target_size).fill_(-10000.)
        else:
            init_alphas = torch.FloatTensor(
                bsz, self.target_size).fill_(-10000.)
        init_alphas[:, Config.START].fill_(0.)
        forward_var = init_alphas

        feats_t = feats.transpose(0, 1)
        for words in feats_t:
            alphas_t = []
            for next_tag in range(self.target_size):
                emit_score = words[:, next_tag].view(-1, 1)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(self.log_sum_exp(next_tag_var, True))
            forward_var = torch.cat(alphas_t, dim=-1)
        forward_var = forward_var + self.transitions[Config.STOP].view(
            1, -1)

        return self.log_sum_exp(forward_var)

    def _get_lstm_features(self, sentences, lengths):
        embeds = self.char_embedds(sentences)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True)
        packed_out, _ = self.lstm(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True)
        lstm_feats = self.hidden2tag(output)

        return lstm_feats

    def _score_sentence(self, feats, tags):
        bsz, sent_len, l_size = feats.size()
        if Config.use_gpu:
            score = torch.cuda.FloatTensor(bsz).fill_(0.)
            s_score = torch.cuda.LongTensor([[Config.START]] * bsz)
        else:
            score = torch.FloatTensor(bsz).fill_(0.)
            s_score = torch.LongTensor([[Config.START]] * bsz)

        tags = torch.cat([s_score, tags], dim=-1)
        feats_t = feats.transpose(0, 1)

        for i, words in enumerate(feats_t):
            temp = self.transitions.index_select(1, tags[:, i])
            bsz_t = self.gather_index(temp.transpose(0, 1), tags[:, i + 1])
            w_step_score = self.gather_index(words, tags[:, i + 1])
            score = score + bsz_t + w_step_score

        temp = self.transitions.index_select(1, tags[:, -1])
        if Config.use_gpu:
            bsz_t = self.gather_index(temp.transpose(0, 1),
                                      (torch.cuda.LongTensor([Config.STOP] * bsz)))
        else:
            bsz_t = self.gather_index(temp.transpose(0, 1),
                                      (torch.LongTensor([Config.STOP] * bsz)))
        return score + bsz_t

    def _viterbi_decode(self, feats):
        backpointers = []
        bsz, sent_len, l_size = feats.size()

        if Config.use_gpu:
            init_vvars = torch.cuda.FloatTensor(
                bsz, self.target_size).fill_(-10000.)
        else:
            init_vvars = torch.FloatTensor(
                bsz, self.target_size).fill_(-10000.)
        init_vvars[:, Config.START].fill_(0.)
        forward_var = init_vvars

        feats_t = feats.transpose(0, 1)
        for words in feats_t:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.target_size):
                _trans = self.transitions[next_tag].view(
                    1, -1).expand_as(words)
                next_tag_var = forward_var + _trans
                best_tag_scores, best_tag_ids = torch.max(
                    next_tag_var, 1, keepdim=True)  # bsz
                bptrs_t.append(best_tag_ids)
                viterbivars_t.append(best_tag_scores)

            forward_var = torch.cat(viterbivars_t, -1) + words
            backpointers.append(torch.cat(bptrs_t, dim=-1))

        terminal_var = forward_var + self.transitions[Config.STOP].view(1, -1)
        _, best_tag_ids = torch.max(terminal_var, 1)

        best_path = [best_tag_ids.view(-1, 1)]
        for bptrs_t in reversed(backpointers):
            best_tag_ids = self.gather_index(bptrs_t, best_tag_ids)
            best_path.append(best_tag_ids.contiguous().view(-1, 1))

        best_path.pop()
        best_path.reverse()

        return torch.cat(best_path, dim=-1)

    def neg_log_likehood(self, sentences, tags, lengths):
        feats = self._get_lstm_features(sentences, lengths)
        forward_score = self._forward_alg(feats)
        tags = tags[:, :torch.max(lengths)]
        gold_score = self._score_sentence(feats, tags)

        return (forward_score-gold_score).sum()

    def forward(self, sentences, lengths):
        lstm_feats = self._get_lstm_features(sentences, lengths)
        tag_seq = self._viterbi_decode(lstm_feats)

        return tag_seq
