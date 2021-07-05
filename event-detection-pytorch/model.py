# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
from utils import init_embedding, init_linear
from torch import einsum

class ACE2019(nn.Module):
    
    def __init__(self, size_embeddings, size_position_embeddings, window_length, number_words,
                 number_positions, number_labels, embeddings=None, use_gpu=True):
        
        super().__init__()

        self.position_embeds = nn.Embedding(number_positions, size_position_embeddings)
#        init_embedding(self.position_embeds.weight)
#        self.position_embeds.weight.data.uniform_(-.5, .5)
        nn.init.orthogonal_(self.position_embeds.weight)

        self.word_embeds = nn.Embedding(number_words, size_embeddings)
        self.use_gpu = use_gpu
        
        if embeddings is not None:
            self.pre_word_embeds = True
            if self.use_gpu:
                self.word_embeds.weight = nn.Parameter(torch.cuda.FloatTensor(embeddings))
            else:
                self.word_embeds.weight = nn.Parameter(
                    torch.FloatTensor(embeddings))
        else:
            self.pre_word_embeds = False
#            init_embedding(self.word_embeds.weight)
#            self.word_embeds.weight.data.uniform_(-.5, .5)
            nn.init.orthogonal_(self.word_embeds.weight)
            
        if self.use_gpu:
            self.word_embeds = self.word_embeds.cuda()
            self.position_embeds = self.position_embeds.cuda()

        self.dropout = nn.Dropout(0.5)

        ngram_filters = [2, 4, 6]
#        ngram_filters = [1, 2, 3]
        filters = [300] * 10
        self.convs = []
        for kernel_size, filter_length, padding in zip(ngram_filters, filters, [0, 0, 1]):
            padding = kernel_size // 2
            conv = nn.Conv1d(in_channels=window_length,#size_embeddings+size_position_embeddings, 
                             out_channels=filter_length, 
                             kernel_size=kernel_size,
                             padding=padding)
            if self.use_gpu:
                conv = conv.cuda()
            
            nn.init.orthogonal_(conv.weight)
            self.convs.append(conv)
        
        max_pool_size = 1
        self.max_pool = nn.AdaptiveMaxPool1d(max_pool_size)
#        self.fc = nn.Linear(filters[0]*max_pool_size, number_labels)
        self.out = nn.Linear(900, number_labels)
#        nn.init.orthogonal_(self.out.weight)
#        self.fc = nn.Linear(filters[0]*114, number_labels)
#        init_linear(self.fc)
        if self.use_gpu:
            self.out = self.out.cuda()
#        self.fc = nn.Linear(num_filters*len(kernel_sizes), number_labels)
#    
    def forward(self, sentence, positions):
        embedded_sent = self.word_embeds(sentence)
        embedded_dist = self.position_embeds(positions)
        embedded_merged = torch.cat([embedded_sent, embedded_dist], dim=-1)
#        import pdb;pdb.set_trace()

#        embedded_merged = embedded_merged.transpose(1,2)  # needs to convert x to (batch, embedding_dim, sentence_len)

        conv_concat = []
        for conv in self.convs:
            conv_x = F.relu(conv(embedded_merged))
            conv_concat.append(conv_x)

#        x = F.relu(torch.cat(conv_concat, dim=-1))
#        import pdb;pdb.set_trace()
#        conv_concat = torch.cat(conv_concat, dim=1)
#        conv_concat = conv_concat.transpose(1,2)
#        x = torch.mean(x, 1) # global avg pool
#        import p
        x = torch.squeeze(self.max_pool(torch.cat(conv_concat, dim=1)), -1)
#        x, _ = torch.max(conv_concat, 1) # global max pool
        x = self.dropout(x)
#        max_pool, _ = torch.max(h_gru, 1)
#        x = F.relu(conv_concat)
#        x = torch.squeeze(self.max_pool(torch.cat(conv_concat, dim=1)), -1)
#        x = x.view(x.size(0), -1)
        return F.softmax(self.out(x), dim=1)
    
class CNN(nn.Module):
#    def __init__(self, kernel_sizes=[3,4,5], num_filters=100, embedding_dim=300, pretrained_embeddings=None):
    def __init__(self, size_embeddings, size_position_embeddings, window_length, number_words,
                 number_positions, number_labels, embeddings=None, use_gpu=True):
        
        super().__init__()
        
        num_filters=100
        embedding_dim=300
        kernel_sizes = [3,4,5]

        self.embedding = nn.Embedding(number_words, embedding_dim)
#        self.embedding.weight.data.uniform_(-.5, .5)
        self.embedding.weight = nn.Parameter(torch.cuda.FloatTensor(embeddings))
#        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
#        self.embedding.weight.requires_grad = mode=="nonstatic"
        
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.embedding = self.embedding.cuda()

        conv_blocks = []
        ConvMethod = "in_channel__is_embedding_dim"
        for kernel_size in kernel_sizes:
            # maxpool kernel_size must <= sentence_len - kernel_size+1, otherwise, it could output empty
            maxpool_kernel_size = window_length - kernel_size +1

            if ConvMethod == "in_channel__is_embedding_dim":
                conv1d = nn.Conv1d(in_channels = embedding_dim, out_channels = num_filters, kernel_size = kernel_size, stride = 1)
            else:
                conv1d = nn.Conv1d(in_channels = 1, out_channels = num_filters, kernel_size = kernel_size*embedding_dim, stride = embedding_dim)

            component = nn.Sequential(
                conv1d,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size = maxpool_kernel_size)
            )
            if self.use_gpu:
                component = component.cuda()

            conv_blocks.append(component)

#            if 0:
#                conv_blocks.append(
#                nn.Sequential(
#                    conv1d,
#                    nn.ReLU(),
#                    nn.MaxPool1d(kernel_size = maxpool_kernel_size)
#                ).cuda()
#                )

        self.conv_blocks = nn.ModuleList(conv_blocks)   # ModuleList is needed for registering parameters in conv_blocks
        self.fc = nn.Linear(num_filters*len(kernel_sizes), number_labels)
        if self.use_gpu:
            self.fc = self.fc.cuda()
            
    def forward(self, sentence, positions):
        
        x = self.embedding(sentence)   # embedded x: (batch, sentence_len, embedding_dim)
        ConvMethod = "in_channel__is_embedding_dim"
        if ConvMethod == "in_channel__is_embedding_dim":
            #    input:  (batch, in_channel=1, in_length=sentence_len*embedding_dim),
            #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
            x = x.transpose(1,2)  # needs to convert x to (batch, embedding_dim, sentence_len)
        else:
            #    input:  (batch, in_channel=embedding_dim, in_length=sentence_len),
            #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
            x = x.view(x.size(0), 1, -1)  # needs to convert x to (batch, 1, sentence_len*embedding_dim)

        x_list= [conv_block(x) for conv_block in self.conv_blocks]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
#        feature_extracted = out
#        out = F.dropout(out, p=0.5, training=self.training)
        return F.softmax(self.fc(out), dim=1)#, feature_extracted

    
import torch.nn as nn
import torch.nn.init as init

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class AttentionConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, dk, dv, num_heads, kernel_size, padding, rel_encoding=True, height=None, width=None):
        super(AttentionConv2d, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dk = dk
        self.dv = dv
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.dkh = self.dk // self.num_heads
        if rel_encoding and not height:
            raise("Cannot use relative encoding without specifying input's height and width")
        self.H = height
        self.W = width

        self.conv_qkv = nn.Conv2d(input_dim, 2*dk + dv, 1)
        self.conv_attn = nn.Conv2d(dv, dv, 1)
        self.conv_out = nn.Conv2d(input_dim, output_dim - dv, kernel_size, padding=padding)
        self.softmax = nn.Softmax(dim=-1)
        self.key_rel_w = nn.Parameter(self.dkh**-0.5 + torch.rand(2*width-1, self.dkh), requires_grad=True)
        self.key_rel_h = nn.Parameter(self.dkh**-0.5 + torch.rand(2*height-1, self.dkh), requires_grad=True)
        self.relative_encoding = rel_encoding
        

    def forward(self, input):
        conv_out = self.conv_out(input)

        qkv = self.conv_qkv(input)    # batch_size, 2*dk+dv, H, W
        
        q, k, v = torch.split(qkv, [self.dk, self.dk, self.dv], dim=1)
        
        batch_size, _, H, W = q.size()

        q = q.view([batch_size, self.num_heads, self.dk // self.num_heads, H*W])
        k = k.view([batch_size, self.num_heads, self.dk // self.num_heads, H*W])
        v = v.view([batch_size, self.num_heads, self.dv // self.num_heads, H*W])

        q *= self.dkh ** -0.5
        logits = einsum('ijkl, ijkm -> ijlm', q, k)
        if self.relative_encoding:
            h_rel_logits, w_rel_logits = self._relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits

        weights = self.softmax(logits)
        attn_out = einsum('ijkl, ijfl -> ijfk', weights, v)
        attn_out = attn_out.contiguous().view(batch_size, self.dv, H, W)
        attn_out = self.conv_attn(attn_out)
        output = torch.cat([conv_out, attn_out], dim=1)
        return output

    def _relative_logits(self, q):
        b, nh, dkh, _ = q.size()
        q = q.view(b, nh, dkh, self.H, self.W)

        rel_logits_w = self._relative_logits1d(q, self.key_rel_w, self.H, self.W, nh, [0, 1, 2, 4, 3, 5])
        rel_logits_h = self._relative_logits1d(q.permute(0, 1, 2, 4, 3), self.key_rel_h, self.W, self.H, nh, [0, 1, 4, 2, 5, 3])
        return rel_logits_h, rel_logits_w

    def _relative_logits1d(self, q, rel_k, H, W, Nh, transpose_mask):
        rel_logits = einsum('bhdxy, md -> bhxym', q, rel_k)

        rel_logits = rel_logits.view([-1, Nh*H, W, 2*W-1])
        rel_logits = self._rel_to_abs(rel_logits)
        rel_logits = rel_logits.view([-1, Nh, H, W, W]).unsqueeze(dim=3).repeat([1,1,1,H,1,1])
        rel_logits = rel_logits.permute(*transpose_mask)
        rel_logits = rel_logits.contiguous().view(-1, Nh, H*W, H*W)
        return rel_logits

    def _rel_to_abs(self, x):
        b, nh, l, _ = x.size()


        x = F.pad(x, (0,1), 'constant', 0)
        flat_x = x.view([b, nh, l*(2*l)]);
        flat_x_padded = F.pad(flat_x, (0, l-1), 'constant', 0)

        final_x = flat_x_padded.view([b, nh, l+1, 2*l-1])
        final_x = final_x[:, :, :l, l-1:]

        return final_x

class BasicAttentionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, height, width, dk, dv, dropRate=0.0):
        super(BasicAttentionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)

        #self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = AttentionConv2d(in_planes, out_planes, height, width, dk, dv, num_heads=8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, height, width, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, height, width, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, height, width, dropRate):
        layers = []
        dk = int(0.1 * out_planes)
        dv = int(0.2 * out_planes)

        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, height, width, dk, dv, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class CNN_with_chars(nn.Module):
  
    def __init__(self, size_embeddings, window_length, number_chars,
                 number_labels, embeddings=None, use_gpu=True):
        
        super().__init__()
        kernel_sizes = [1, 2, 3, 4]
        filter_sizes = [300] * 10

        self.char_embeds = nn.Embedding(number_chars, size_embeddings, padding_idx=0).cuda()
        nn.init.orthogonal_(self.char_embeds.weight)

        self.conv_0 = nn.Conv2d(in_channels = 1,
                                out_channels = filter_sizes[0], 
                                kernel_size = (kernel_sizes[0], size_embeddings)).cuda()
        
        self.conv_1 = nn.Conv2d(in_channels = 1, 
                                out_channels = filter_sizes[1], 
                                kernel_size = (kernel_sizes[1], size_embeddings)).cuda()
        
        self.conv_2 = nn.Conv2d(in_channels = 1, 
                                out_channels = filter_sizes[2], 
                                kernel_size = (kernel_sizes[2], size_embeddings)).cuda()

        self.fc = nn.Linear(900, number_labels).cuda()
        
        self.dropout = nn.Dropout(.5)
        self.dropout1 = nn.Dropout(.3)
        self.norm = L2Norm(1200, 3)
        
    def forward(self, text):
                
        #text = [batch size, sent len]
        embedded_chars = self.char_embeds(text)
        
        embedded_chars = embedded_chars.unsqueeze(1)
        embedded_chars = self.dropout(embedded_chars)

        conved_0 = torch.tanh(self.conv_0(embedded_chars).squeeze(3))
        conved_1 = torch.tanh(self.conv_1(embedded_chars).squeeze(3))
        conved_2 = torch.tanh(self.conv_2(embedded_chars).squeeze(3))
        
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))

        return self.fc(cat)
      
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=31):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
#        import pdb;pdb.set_trace()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)#.transpose(0,1)
        pe[:, 1::2] = torch.cos(position * div_term)#.transpose(0,1)
        
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
#        import pdb;pdb.set_trace()
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim 
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class Attention_Net(nn.Module):

    def __init__(self, size_embeddings, size_position_embeddings, window_length, number_words,
                 number_positions, number_labels, embeddings=None, use_gpu=True):

        super(Attention_Net, self).__init__()
#        drp = 0.1
        self.embedding = nn.Embedding(number_words, size_embeddings).cuda()
#        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
#        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(size_embeddings, 128, bidirectional=True, batch_first=True).cuda()
        self.lstm2 = nn.GRU(128*2, 64, bidirectional=True, batch_first=True).cuda()

        self.attention_layer = Attention(128, window_length).cuda()
        
        self.linear = nn.Linear(64*2 , 64).cuda()
        self.relu = nn.ReLU()
        print('number labels', number_labels)
        self.out = nn.Linear(64, 1).cuda()

    def forward(self, text, positions):
        h_embedding = self.embedding(text)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        h_lstm, _ = self.lstm(h_embedding)
        h_lstm, _ = self.lstm2(h_lstm)
        h_lstm_atten = self.attention_layer(h_lstm)
        conc = self.relu(self.linear(h_lstm_atten))
#        print(conc.shape)
        
        out = self.out(conc)
        return out

from  torch.nn.modules.transformer import TransformerEncoderLayer
from torch.autograd import Variable
class CNN_with_words(nn.Module):
#    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
#                 dropout, pad_idx):
  
    def __init__(self, size_embeddings, size_position_embeddings, window_length, number_words,
                 number_positions, number_labels, embeddings=None, use_gpu=True):
        
        super().__init__()
        kernel_sizes = [1, 2, 3, 4]
#        kernel_sizes = [2, 3, 4, 5]
        filter_sizes = [300] * 10
        
        self.word_embeds = nn.Embedding(number_words, size_embeddings, padding_idx=0).cuda()
#        nn.init.orthogonal_(self.word_embeds.weight)
#        nn.init.uniform_(self.word_embeds.weight)
        self.word_embeds.weight.data.uniform_(-.5, .5)
#        self.word_embeds.weight = nn.Parameter(torch.cuda.FloatTensor(embeddings))
        self.position_embeds = nn.Embedding(number_positions, size_position_embeddings, padding_idx=0).cuda()
        nn.init.orthogonal_(self.position_embeds.weight)
        
        self.pos_encoder = PositionalEncoding(300, .1).cuda()
        
#        import pdb;pdb.set_trace()
        self.conv_0 = nn.Conv2d(in_channels = 1,
                                out_channels = filter_sizes[0], 
                                kernel_size = (kernel_sizes[0], size_embeddings)).cuda()
#                                kernel_size = (kernel_sizes[0], size_embeddings + size_position_embeddings)).cuda()
        
        self.conv_1 = nn.Conv2d(in_channels = 1,
                                out_channels = filter_sizes[1], 
                                kernel_size = (kernel_sizes[1], size_embeddings)).cuda()
#                                kernel_size = (kernel_sizes[1], size_embeddings + size_position_embeddings)).cuda()
        
        self.conv_2 = nn.Conv2d(in_channels = 1,
                                out_channels = filter_sizes[2], 
                                kernel_size = (kernel_sizes[2], size_embeddings)).cuda()
#                                kernel_size = (kernel_sizes[2], size_embeddings + size_position_embeddings)).cuda()
#        
#        self.conv_3 = nn.Conv2d(in_channels = 1, 
#                                out_channels = filter_sizes[3], 
#                                kernel_size = (kernel_sizes[3], size_embeddings+size_position_embeddings)).cuda()

        self.batch_norm = nn.BatchNorm1d(filter_sizes[2]).cuda()
        
        self.lstm = nn.LSTM(300, 256).cuda()

        self.fc = nn.Linear(256, number_labels).cuda()
        
        self.dropout = nn.Dropout(.5)
        self.dropout1 = nn.Dropout(.3)
        self.norm = L2Norm(filter_sizes[0]*4, 3)
        self.ninp = 300
        self.batch_size = 64
        self.hidden_dim = 256

        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        if True:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)
    
    def forward(self, text, positions):
                
        #text = [batch size, sent len]
        embedded_words = self.word_embeds(text)
#        print(self.word_embeds(text).shape)
        embedded_positions = self.position_embeds(positions)
        
        embedded_words = embedded_words.unsqueeze(1)
        embedded_words = self.dropout(embedded_words)
        embedded_positions = embedded_positions.unsqueeze(1)
        
#        print(embedded_words.shape)
#        print(embedded_words[:5])

#        embedded = torch.cat([embedded_words, embedded_positions], dim=-1)

#        embedded = embedded_words * math.sqrt(self.ninp)
#        embedded = self.pos_encoder(embedded)
#        
##        cat = self.lstm(cat)
#        print(cat.shape)
#        
#        cat = cat.squeeze(1)
#
#        embedded = embedded.contiguous().view(embedded.shape[0], -1)
        embedded_words = self.word_embeds(text)
#        cat = self.lstm(embedded_words)
        
#        print(text.shape, embedded_words.shape)
#        torch.Size([64, 31]) torch.Size([64, 31, 300])
#        torch.Size([31, 64, 300])

        cat = embedded_words.view(text.shape[1], embedded_words.shape[0], -1)
#        print(cat.shape)
        
        lstm_out, self.hidden = self.lstm(cat, self.hidden)
        out  = self.fc(lstm_out[-1])
        
#        print(cat.shape)
        
#        
#        conved_0 = torch.tanh(self.conv_0(embedded).squeeze(3))
#        conved_1 = torch.tanh(self.conv_1(embedded).squeeze(3))
#        conved_2 = torch.tanh(self.conv_2(embedded).squeeze(3))
##        conved_3 = torch.tanh(self.conv_3(embedded).squeeze(3))
##        conved_2 = self.batch_norm(conved_2)
#
##        pooled_0, _ = torch.max(conved_0, 2)
##        pooled_1, _ = torch.max(conved_1, 2)
##        pooled_2, _ = torch.max(conved_2, 2)
##        pooled_3, _ = torch.max(conved_3, 2)
#        
#        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
#        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
#        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
#        pooled_3 = F.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)

        
#        cat = src + self.dropout1(src2)
#torch.Size([64, 1, 31, 300])
#torch.Size([64, 1, 31, 350])
#torch.Size([64, 900])
#scores torch.Size([64, 34])

        #pooled_n = [batch size, n_filters]
#        import pdb;pdb.set_trace()
#        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))
#        print(cat.shape)
        
#        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2, pooled_3), dim = 1))
#        cat = self.dropout1(torch.cat((pooled_0, pooled_1), dim=1))
#        out = self.fc(cat)
        
        return out
      
      
import math
class CNN_with_attention(nn.Module):

    def __init__(self, size_embeddings, size_position_embeddings, window_length, number_words,
                 number_positions, number_labels, embeddings=None, use_gpu=True):


#    def __init__(self, depth, num_classes, widen_factor=1, input_dim=(32, 32), dropRate=0.0):
        super(CNN_with_attention, self).__init__()
        widen_factor = 1
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        depth = 6
        dropRate=0.0

        height, width = number_words, size_embeddings

        self.word_embeds = nn.Embedding(number_words, size_embeddings, padding_idx=0).cuda()
        nn.init.orthogonal_(self.word_embeds.weight)
        
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicAttentionBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, height, width, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, height, width, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, height, width, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], number_labels)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, text, positions):
                
        #text = [batch size, sent len]
        embedded_words = self.word_embeds(text)
        embedded_positions = self.position_embeds(positions)
        
        embedded_words = embedded_words.unsqueeze(1)
        embedded_words = self.dropout(embedded_words)
        embedded_positions = embedded_positions.unsqueeze(1)

        embedded = torch.cat([embedded_words, embedded_positions], dim=-1)

        out = self.conv1(embedded)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        
        out = F.avg_pool2d(out, 32)
        out = out.view(-1, self.nChannels)
        #print(out.size())
        return self.fc(out)
    
