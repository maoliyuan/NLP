#coding=utf8
import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import copy

def find_best_match(value, sentence): # value=ontology sentence=predicted sentence when inferencing
    bidx = 0
    best_score = 0
    min_length = min(len(value), len(sentence))
    for idx in range(len(sentence)-min_length+1):
        pres_score = len(np.where(np.array(list(value)[: min_length])==np.array(list(sentence)[idx: idx+min_length]))[0])
        (best_score, bidx) =  (pres_score, idx) if pres_score > best_score else (best_score, bidx)
    return bidx, best_score / min_length

def map(tensor):# 0->PAD 1->O 2->B 3->I
    BIO_tensor = copy.copy(tensor)
    BIO_tensor[(tensor >=2) * (tensor % 2 == 0)] = 2
    BIO_tensor[(tensor >=3) * (tensor % 2 == 1)] = 3
    slot_tensor = torch.where(tensor == 1, 0, tensor)
    slot_tensor =torch.where(slot_tensor > 0, torch.ceil((tensor-1) / 2), slot_tensor)
    return BIO_tensor, slot_tensor.to(dtype=torch.int64)

def unmap(BIO_tensor, slot_tensor):
    tensor = torch.where(BIO_tensor == 2, 2 * slot_tensor, BIO_tensor)
    tensor = torch.where(BIO_tensor == 3, 2 * slot_tensor + 1, tensor)
    return tensor

class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.observed_values_path = os.path.join(config.dataroot, 'observed_values.txt')
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.BI_embedding_dim = config.BI_embedding_dim
        self.BIO_tagger = TaggingFNNDecoder(config.hidden_size, 4, config.tag_pad_idx)
        self.slot_tagger = TaggingFNNDecoder(config.hidden_size + self.BI_embedding_dim, 37, config.tag_pad_idx)

        with open(self.observed_values_path, 'r') as f:
            self.value_list = f.readlines()

    def forward(self, batch):
        tag_ids = batch.tag_ids
        BIO_ids, slot_ids = map(tag_ids)
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths
        embed = self.word_embed(input_ids)
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        hiddens = self.dropout_layer(rnn_out)
        BIO_prob, BIO_loss = self.BIO_tagger(hiddens, tag_mask, BIO_ids)
        BIO_label = torch.argmax(BIO_prob, dim=-1)

        B_embedding = torch.zeros([hiddens.shape[0], hiddens.shape[1], self.BI_embedding_dim]).to(self.config.device)
        B_embedd_hidden = torch.cat([hiddens, B_embedding], dim=-1)
        I_embedding = torch.ones([hiddens.shape[0], hiddens.shape[1], self.BI_embedding_dim]).to(self.config.device)
        I_embedd_hidden = torch.cat([hiddens, I_embedding], dim=-1)
        B_embedd_hidden[BIO_ids == 3] = I_embedd_hidden[BIO_ids == 3]
        slot_prob, slot_loss = self.slot_tagger(B_embedd_hidden, tag_mask, slot_ids)
        slot_label = torch.argmax(slot_prob, dim=-1)

        tag_output = unmap(BIO_label, slot_label)
        total_loss = BIO_loss + slot_loss
        return tag_output, total_loss

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        tag_output, loss = self.forward(batch)
        predictions = []
        for i in range(batch_size):
            pred = tag_output[i].cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:

                    possible_slots = {}
                    for labeled_word in tag_buff:
                        slot = '-'.join(labeled_word.split('-')[1:])
                        possible_slots[slot] = possible_slots.get(slot, 0) + 1
                    possible_slots = list(possible_slots.items())
                    possible_slots.sort(key = lambda x: x[1], reverse = True)

                    init_value = ''.join([batch.utt[i][j] for j in idx_buff])
                    final_value = init_value
                    best_score = 0
                    for observed_value in self.value_list:
                        _, score = find_best_match(observed_value, init_value)
                        if score > best_score:
                            final_value, best_score = observed_value, score 
                    
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{possible_slots[0][0]}-{final_value}')

                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                possible_slots = {}
                for labeled_word in tag_buff:
                    slot = '-'.join(labeled_word.split('-')[1:])
                    possible_slots[slot] = possible_slots.get(slot, 0) + 1
                possible_slots = list(possible_slots.items())
                possible_slots.sort(key = lambda x: x[1], reverse = True)

                init_value = ''.join([batch.utt[i][j] for j in idx_buff])
                final_value = init_value
                best_score = 0
                for observed_value in self.value_list:
                    _, score = find_best_match(observed_value, init_value)
                    if score > best_score:
                        final_value, best_score = observed_value, score 
                
                pred_tuple.append(f'{possible_slots[0][0]}-{final_value}')
            predictions.append(pred_tuple)
        return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, num_tags)
        self.relu = nn.ReLU()
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        h1 = self.relu(self.fc1(hiddens))
        h2 = self.relu(self.fc2(h1))
        logits = self.output_layer(h2)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return prob
