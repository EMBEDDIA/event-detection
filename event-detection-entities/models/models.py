# -*- coding: utf-8 -*-

from fastNLP.modules import ConditionalRandomField, allowed_transitions
from modules.transformer import TransformerEncoder, MultiHeadAttn, TransformerLayer
from torch import nn
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from fastNLP.core.const import Const as C
from fastNLP.modules.encoder.lstm import LSTM
from fastNLP.embeddings.utils import get_embeddings
from fastNLP.modules.decoder.mlp import MLP


class BiLSTMSentiment(nn.Module):
    def __init__(self, init_embed,
                 num_classes,
                 hidden_dim=256,
                 num_layers=1,
                 nfc=128):
        super(BiLSTMSentiment,self).__init__()
        self.embed = get_embeddings(init_embed)
        self.lstm = LSTM(input_size=self.embed.embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True)
        self.mlp = MLP(size_layer=[hidden_dim*2, nfc, num_classes])

    def forward(self, words):
        x_emb = self.embed(words)
        output, _ = self.lstm(x_emb)
        output = self.mlp(torch.max(output, dim=1)[0])
        return {C.OUTPUT: output}

    def predict(self, words):
        output = self(words)
        _, predict = output[C.OUTPUT].max(dim=1)
        return {C.OUTPUT: predict}

class StackedTransformersCRF(nn.Module):
    def __init__(self, tag_vocabs, embed, embed_doc, num_layers, d_model, n_head, feedforward_dim, dropout,
                 after_norm=True, attn_type='adatrans',  bi_embed=None,
                 fc_dropout=0.3, pos_embed=None, scale=False, dropout_attn=None):

        super().__init__()

        self.embed = embed
        self.embed_doc = embed_doc
        
        embed_size = self.embed.embed_size
        self.bi_embed = None
        if bi_embed is not None:
            self.bi_embed = bi_embed
            embed_size += self.bi_embed.embed_size

        self.tag_vocabs = []
        self.out_fcs = nn.ModuleList()
        self.crfs = nn.ModuleList()
        
        for i in range(len(tag_vocabs)):
            self.tag_vocabs.append(tag_vocabs[i])
            
#            linear = nn.Linear(768, len(tag_vocabs[i]))
            linear = nn.Linear(1536, len(tag_vocabs[i]))
#            linear = nn.Linear(1792, len(tag_vocabs[i]))

            self.out_fcs.append(linear)
            trans = allowed_transitions(
                tag_vocabs[i], encoding_type='bioes', include_start_end=True)
            crf = ConditionalRandomField(
                len(tag_vocabs[i]), include_start_end_trans=True, allowed_transitions=trans)
            self.crfs.append(crf)
            
        
        self.in_fc = nn.Linear(embed_size, d_model)
        self.in_fc_doc = nn.Linear(embed_size, d_model)
#        self.in_fc_doc_flip = nn.Linear(d_model, d_model)

        self.transformer = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)

#        n_heads = 12# 
#        head_dims = 128
        self.transformer_doc = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
        self.self_attn = MultiHeadAttn(d_model, n_head)
        
        hidden_dim = 512
#        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        self.lstm = LSTM(input_size=self.embed_doc.embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True)
        
        self.pooling_methods = ['max', 'mean', 'max-mean']
        
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc_dropout_doc = nn.Dropout(fc_dropout)
        
    def _mean_pooler(self, encoding):
        return encoding.mean(dim=1)
    
    def _max_pooler(self, encoding):
        return encoding.max(dim=1).values
    
    def _max_mean_pooler(self, encoding):
        return torch.cat((self._max_pooler(encoding), self._mean_pooler(encoding)), dim=1)
    
    def _pooler(self, encodings, pooling_method):
        '''
        Pools the encodings along the time/sequence axis according
        to one of the pooling method:
            - 'max'      :  max value along the sequence/time dimension
                            returns a (batch_size x hidden_size) shaped tensor
            - 'mean'     :  mean of the values along the sequence/time dimension
                            returns a (batch_size x hidden_size) shaped tensor
            - 'max-mean' :  max and mean values along the sequence/time dimension appended
                            returns a (batch_size x 2*hidden_size) shaped tensor
                            [ max : mean ]
        Parameters
        ----------
        encoding : list of tensor to pool along the sequence/time dimension.
        
        pooling_method : one of 'max', 'mean' or 'max-mean'
        
        Returns
        -------
        tensor of shape (batch_size x hidden_size).
        '''
        
        assert (pooling_method in self.pooling_methods), \
            "pooling methods needs to be one of 'max', 'mean' or 'max-mean'"
            
        if pooling_method   == 'max':       pool_fn = self._max_pooler
        elif pooling_method == 'mean':      pool_fn = self._mean_pooler
        elif pooling_method == 'max-mean':  pool_fn = self._max_mean_pooler
        
        pooled = pool_fn(encodings)
        
        return pooled
    
    def _forward(self, words, doc=None, target=None, target1=None, target2=None, target3=None, target4=None, target5=None, target6=None, bigrams=None, seq_len=None):

        torch.cuda.empty_cache()

#        print(doc.shape, words.shape)

        mask = words.ne(0)
        words = self.embed(words)
        
#        print('!!!!!!!!!!!!!!', doc.shape, max(doc[0]), max(doc[1]))
##        
#        if doc.shape[1] >= 512:
#            import pdb;pdb.set_trace()
#            doc = torch.split(doc, 510, dim=1)[0]
        
#        mask_doc = doc.ne(0)
#        doc = self.embed_doc(doc)

#        if self.bi_embed is not None:
#            bigrams = self.bi_embed(bigrams)
#            words = torch.cat([words, bigrams], dim=-1)
        
        torch.cuda.empty_cache()
        #targets = [target, target1, target2, target3, target4, target5, target6]
#        targets = [target, target1]
        targets = [target]
        
#        print('words', words.shape)
        chars = self.in_fc(words)
        
#        print('chars', chars.shape)
        chars = self.transformer(chars, mask)
#        print('chars', chars.shape)

        words = self.fc_dropout(chars)
#        print('words', words.shape)
#        torch.cuda.empty_cache()
#        
#        chars_doc = self.in_fc_doc(doc)
#        chars_doc = self.self_attn(chars_doc, mask_doc)
#        chars_doc = self.fc_dropout_doc(chars_doc)
##        v = torch.matmul(attn, v)
##        chars_doc = self.transformer_doc(chars_doc, mask_doc)
##        import pdb;pdb.set_trace()
#        
#        import pdb;pdb.set_trace()
#        output, _ = self.lstm(doc)
#        
##        lstm_doc = torch.mean(output, dim=1)[0]
#        lstm_doc = torch.mean(output, dim=1)
#        lstm_doc_repeat = lstm_doc.unsqueeze(1).repeat(1, words.shape[1], 1)
#        words = torch.cat([words, lstm_doc_repeat], -1)
        
#        doc = self.in_fc_doc(doc)
        
#        pooled_doc = self._pooler(doc, 'max')
#        pooled_doc_repeat = pooled_doc.unsqueeze(1).repeat(1, words.shape[1], 1)
#        words = torch.cat([words, pooled_doc_repeat], -1)
#        import pdb;pdb.set_trace()
##        pooled_output = pooled_output.view((pooled_output.shape[0], words.shape[1], -1))
##        import pdb;pdb.set_trace()
##        words = torch.cat([words, pooled_output], 2)
#        words = torch.add(pooled_output.unsqueeze(1), words)
        
#        pool = nn.AdaptiveAvgPool2d((chars.shape[1], chars.shape[2]))
        
#        chars = torch.einsum('ikm, jkm-> ikm', chars, pool(chars_doc))
        
#        chars = torch.add(torch.mean(chars_doc, 1).unsqueeze(1), chars)
        
#        chars = torch.add(torch.mean(chars_doc, 1).unsqueeze(1), chars)
        
        #chars = self.fc_dropout(chars)
        
#        chars_doc = torch.cat(chars.shape[1] * [chars_doc.unsqueeze(1)], 1)
#        
#        chars_doc = chars_doc.view(chars.shape[0], chars.shape[1], chars_doc.shape[2] * chars_doc.shape[3])
        
#        chars = torch.cat([chars, chars_doc], 2)
        
        logits = []
        for i in range(len(targets)):
            logits.append(F.log_softmax(self.out_fcs[i](words), dim=-1))

        torch.cuda.empty_cache()

        if target is not None:
            losses = []
            for i in range(len(targets)):
                losses.append(self.crfs[i](logits[i], targets[i], mask))

            return {'loss': sum(losses)}
        else:
            results = {}
            for i in range(len(targets)):
                if i == 0:
                    results['pred'] = self.crfs[i].viterbi_decode(logits[i], mask)[
                        0]
                else:
                    results['pred' + str(i)] = torch.argmax(logits[i], 2)
            return results

    def forward(self, words, doc=None, target=None, target1=None, target2=None, target3=None, target4=None, target5=None, target6=None, seq_len=None):
        return self._forward(words, doc, target, target1, target2, target3, target4, target5, target6, seq_len)

    def predict(self, words, doc=None, seq_len=None):
        return self._forward(words, doc, target=None)


class BertCRF(nn.Module):
    def __init__(self, embed, tag_vocabs, encoding_type='bio'):
        super().__init__()
        self.embed = embed
        self.tag_vocabs = []
        self.fcs = nn.ModuleList()
        self.crfs = nn.ModuleList()

        for i in range(len(tag_vocabs)):
            self.tag_vocabs.append(tag_vocabs[i])
            linear = nn.Linear(self.embed.embed_size, len(tag_vocabs[i]))
            self.fcs.append(linear)
            trans = allowed_transitions(
                tag_vocabs[i], encoding_type=encoding_type, include_start_end=True)
            crf = ConditionalRandomField(
                len(tag_vocabs[i]), include_start_end_trans=True, allowed_transitions=trans)
            self.crfs.append(crf)

    def _forward(self, words, target=None, target1=None, target2=None, target3=None, target4=None, target5=None, target6=None, seq_len=None):
        mask = words.ne(0)
        words = self.embed(words)

        targets = [target]#, target1, target2, target3, target4, target5, target6]

        words_fcs = []
        for i in range(len(targets)):
            words_fcs.append(self.fcs[i](words))

        logits = []
        for i in range(len(targets)):
            logits.append(F.log_softmax(words_fcs[i], dim=-1))

        if target is not None:
            losses = []
            for i in range(len(targets)):
                losses.append(self.crfs[i](logits[i], targets[i], mask))

            return {'loss': sum(losses)}
        else:
            results = {}
            for i in range(len(targets)):
                if i == 0:
                    results['pred'] = self.crfs[i].viterbi_decode(logits[i], mask)[0]
                else:
                    results['pred' + str(i)] = torch.argmax(logits[i], 2)

            return results

    def forward(self, words, target=None, target1=None, target2=None, target3=None, target4=None, target5=None, target6=None, seq_len=None):
        return self._forward(words, target, target1, target2, target3, target4, target5, target6, seq_len)

    def predict(self, words, seq_len=None):
        return self._forward(words, target=None)
