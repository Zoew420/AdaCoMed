"""
This code is developed based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import tfidf_loading
import numpy as np
import pickle
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.distributions.normal import Normal
import functools
import operator

def add_noise(images, mean=0, std=0.1):
    normal_dst = Normal(mean, std)
    noise = normal_dst.sample(images.shape)
    noisy_image = noise + images
    return noisy_image

def print_model(model):
    print(model)
    nParams = 0
    for w in model.parameters():
        nParams += functools.reduce(operator.mul, w.size(), 1)
    print(nParams)

class Auto_Encoder_Model(nn.Module):
    def __init__(self):
        super(Auto_Encoder_Model, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, padding=1, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 32, padding=1, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 16, padding=1, kernel_size=3)

        # Decoder
        self.tran_conv1 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.tran_conv2 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward_pass(self, x):
        output = F.relu(self.conv1(x))
        output = self.max_pool1(output)
        output = F.relu(self.conv2(output))
        output = self.max_pool2(output)
        output = F.relu(self.conv3(output))
        return output

    def reconstruct_pass(self, x):
        output = F.relu(self.tran_conv1(x))
        output = F.relu(self.conv4(output))
        output = F.relu(self.tran_conv2(output))
        output = torch.sigmoid(self.conv5(output))
        return output

    def forward(self, x):
        output = self.forward_pass(x)
        output = self.reconstruct_pass(output)
        return output

class SimpleCNN(nn.Module):
    def __init__(self, weight_path='simple_cnn.weights', eps_cnn=1e-5, momentum_cnn=0.05):
        super(SimpleCNN, self).__init__()
        # init and load pre-trained model
        weights = self.load_weight(weight_path)
        self.conv1 = self.init_conv(1, 64, weights['conv1'], weights['b1'])
        self.conv1_bn = nn.BatchNorm2d(num_features=64, eps=eps_cnn, affine=True, momentum=momentum_cnn)
        self.conv2 = self.init_conv(64, 64, weights['conv2'], weights['b2'])
        self.conv2_bn = nn.BatchNorm2d(num_features=64, eps=eps_cnn, affine=True, momentum=momentum_cnn)
        self.conv3 = self.init_conv(64, 64, weights['conv3'], weights['b3'])
        self.conv3_bn = nn.BatchNorm2d(num_features=64, eps=eps_cnn, affine=True, momentum=momentum_cnn)
        self.conv4 = self.init_conv(64, 64, weights['conv4'], weights['b4'])
        self.conv4_bn = nn.BatchNorm2d(num_features=64, eps=eps_cnn, affine=True, momentum=momentum_cnn)

    def load_weight(self, path):
        return pickle.load(open(path, 'rb'))

    def forward(self, X):
        out = F.relu(self.conv1(X))
        out = self.conv1_bn(out)
        out = F.relu(self.conv2(out))
        out = self.conv2_bn(out)
        out = F.relu(self.conv3(out))
        out = self.conv3_bn(out)
        out = F.relu(self.conv4(out))
        out = self.conv4_bn(out)
        out = out.view(-1, 64, 36)

        return torch.mean(out, 2)

    def convert_to_torch_weight(self, weight):
        return np.transpose(weight, [3, 2, 0, 1])

    def init_conv(self, inp, out, weight, bias, convert=True):
        conv = nn.Conv2d(inp, out, 3, 2, 1, bias=True)
        if convert:
            weight = self.convert_to_torch_weight(weight)
        conv.weight.data = torch.Tensor(weight).float()
        conv.bias.data = torch.Tensor(bias).float()
        return conv

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, args):
        super(SimpleClassifier, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        activation_dict = {'relu': nn.ReLU()}
        try:
            activation_func = activation_dict[args.activation]
        except:
            raise AssertionError(args.activation + " is not supported yet!")
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            activation_func,
            nn.Dropout(args.dropout, inplace=False),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout, op=''):
        super(WordEmbedding, self).__init__()
        self.op = op
        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
        if 'c' in op:
            self.emb_ = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
            self.emb_.weight.requires_grad = False # fixed
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file, tfidf=None, tfidf_weights=None):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init
        if tfidf is not None:
            if 0 < tfidf_weights.size:
                weight_init = torch.cat([weight_init, torch.from_numpy(tfidf_weights)], 0)
            weight_init = tfidf.matmul(weight_init) # (N x N') x (N', F)
            self.emb_.weight.requires_grad = True
        if 'c' in self.op:
            self.emb_.weight.data[:self.ntoken] = weight_init.clone()

    def forward(self, x):
        emb = self.emb(x)
        if 'c' in self.op:
            emb = torch.cat((emb, self.emb_(x)), 2)
        emb = self.dropout(emb)
        return emb

class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU if rnn_type == 'GRU' else None

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)
    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid // self.ndirections)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        output, hidden = self.rnn(x, hidden)
        return output


# Stacked Attention
class StackedAttention(nn.Module):
    def __init__(self, num_stacks, img_feat_size, ques_feat_size, att_size, output_size, drop_ratio):
        super(StackedAttention, self).__init__()

        self.img_feat_size = img_feat_size
        self.ques_feat_size = ques_feat_size
        self.att_size = att_size
        self.output_size = output_size
        self.drop_ratio = drop_ratio
        self.num_stacks = num_stacks
        self.layers = nn.ModuleList()

        self.dropout = nn.Dropout(drop_ratio)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.fc11 = nn.Linear(ques_feat_size, att_size, bias=True)
        self.fc12 = nn.Linear(img_feat_size, att_size, bias=False)
        self.fc13 = nn.Linear(att_size, 1, bias=True)

        for stack in range(num_stacks - 1):
            self.layers.append(nn.Linear(att_size, att_size, bias=True))
            self.layers.append(nn.Linear(img_feat_size, att_size, bias=False))
            self.layers.append(nn.Linear(att_size, 1, bias=True))

    def forward(self, img_feat, ques_feat, v_mask=True):

        # Batch size
        B = ques_feat.size(0)

        # Stack 1
        ques_emb_1 = self.fc11(ques_feat)
        img_emb_1 = self.fc12(img_feat)

        # Compute attention distribution
        h1 = self.tanh(ques_emb_1.view(B, 1, self.att_size) + img_emb_1)
        h1_emb = self.fc13(self.dropout(h1))
        # Mask actual bounding box sizes before calculating softmax
        if v_mask:
            mask = (0 == img_emb_1.abs().sum(2)).unsqueeze(2).expand(h1_emb.size())
            h1_emb.data.masked_fill_(mask.data, -float('inf'))

        p1 = self.softmax(h1_emb)

        #  Compute weighted sum
        img_att_1 = img_emb_1*p1
        weight_sum_1 = torch.sum(img_att_1, dim=1)

        # Combine with question vector
        u1 = ques_emb_1 + weight_sum_1

        # Other stacks
        us = []
        ques_embs = []
        img_embs  = []
        hs = []
        h_embs =[]
        ps  = []
        img_atts = []
        weight_sums = []

        us.append(u1)
        for stack in range(self.num_stacks - 1):
            ques_embs.append(self.layers[3 * stack + 0](us[-1]))
            img_embs.append(self.layers[3 * stack + 1](img_feat))

            # Compute attention distribution
            hs.append(self.tanh(ques_embs[-1].view(B, -1, self.att_size) + img_embs[-1]))
            h_embs.append(self.layers[3*stack + 2](self.dropout(hs[-1])))
            # Mask actual bounding box sizes before calculating softmax
            if v_mask:
                mask = (0 == img_embs[-1].abs().sum(2)).unsqueeze(2).expand(h_embs[-1].size())
                h_embs[-1].data.masked_fill_(mask.data, -float('inf'))
            ps.append(self.softmax(h_embs[-1]))

            #  Compute weighted sum
            img_atts.append(img_embs[-1] * ps[-1])
            weight_sums.append(torch.sum(img_atts[-1], dim=1))

            # Combine with previous stack
            ux = us[-1] + weight_sums[-1]

            # Combine with previous stack by multiple
            us.append(ux)

        return us[-1]

# Create SAN model
class SAN_Model(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, classifier, args, maml_v_emb, ae_v_emb):
        super(SAN_Model, self).__init__()
        self.args = args
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.classifier = classifier
        if args.maml:
            self.maml_v_emb = maml_v_emb
        if args.autoencoder:
            self.ae_v_emb = ae_v_emb
            self.convert = nn.Linear(16384, 64)
    def forward(self, v, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        # get visual feature
        if self.args.maml:
            maml_v_emb = self.maml_v_emb(v[0]).unsqueeze(1)
            v_emb = maml_v_emb
        if self.args.autoencoder:
            encoder = self.ae_v_emb.forward_pass(v[1])
            decoder = self.ae_v_emb.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
            v_emb = ae_v_emb
        if self.args.maml and self.args.autoencoder:
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)
        # get textual feature
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim], return final hidden state
        # Attention
        att = self.v_att(v_emb, q_emb)
        if self.args.autoencoder:
            return att, decoder
        return att

    def classify(self, input_feats):
        return self.classifier(input_feats)

# Build SAN model
def build_SAN(dataset, args):
    # init word embedding module, question embedding module, and Attention network
    ntoken = 1177
    num_ans_candidates = 458
    v_dim = 128
    w_emb = WordEmbedding(ntoken, 300, 0.0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, 0.0, args.rnn)
    v_att = StackedAttention(args.num_stacks, v_dim, args.num_hid, args.num_hid, num_ans_candidates,
                             args.dropout)
    # build and load pre-trained MAML model
    if args.maml:
        weight_path = args.RAD_dir + '/' + args.maml_model_path
        # print('load initial weights MAML from: %s' % (weight_path))
        maml_v_emb = SimpleCNN(weight_path, args.eps_cnn, args.momentum_cnn)
    # build and load pre-trained Auto-encoder model
    if args.autoencoder:
        ae_v_emb = Auto_Encoder_Model()
        weight_path = args.RAD_dir + '/' + args.ae_model_path
        # print('load initial weights DAE from: %s'%(weight_path))
        ae_v_emb.load_state_dict(torch.load(weight_path))
    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args)
    # init classifier
    classifier = SimpleClassifier(
        args.num_hid, 2 * args.num_hid, num_ans_candidates, args)
    # contruct VQA model and return
    if args.maml and args.autoencoder:
        return SAN_Model(w_emb, q_emb, v_att, classifier, args, maml_v_emb, ae_v_emb)
    elif args.maml:
        return SAN_Model(w_emb, q_emb, v_att, classifier, args, maml_v_emb, None)
    elif args.autoencoder:
        return SAN_Model(w_emb, q_emb, v_att, classifier, args, None, ae_v_emb)
    return SAN_Model(w_emb, q_emb, v_att, classifier, args, None, None)