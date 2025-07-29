from ast import Sub
import torch
import torch.nn as nn
import torch.nn.functional as F
from module import *
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super(Attention, self).__init__()
        assert (
            dim % num_heads == 0
        ), "Embedding dimension should be divisible by number of heads"

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C = x.shape
        N = 1
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.reshape(B, C)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate=0.0, revised=False):
        super(FeedForward, self).__init__()
        if not revised:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(dim, hidden_dim, kernel_size=1, stride=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                nn.Conv1d(hidden_dim, dim, kernel_size=1, stride=1),
                nn.BatchNorm1d(dim),
                nn.GELU(),
            )

        self.revised = revised
        self._init_weights()

    def _init_weights(self):
        for name, module in self.net.named_children():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.bias, std=1e-6)

    def forward(self, x):
        if self.revised:
            x = x.permute(0, 2, 1)
            x = self.net(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.net(x)

        return x

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_ratio=4.0,
        attn_dropout=0.0,
        dropout=0.0,
        qkv_bias=True,
        revised=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        assert isinstance(
            mlp_ratio, float
        ), "MLP ratio should be an integer for valid "
        mlp_dim = int(mlp_ratio * dim)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                num_heads=heads,
                                qkv_bias=qkv_bias,
                                attn_drop=attn_dropout,
                                proj_drop=dropout,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout_rate=dropout,),
                        )
                        if not revised
                        else FeedForward(
                            dim, mlp_dim, dropout_rate=dropout, revised=True,
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class OutputLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes=1000,
        representation_size1=None,
        representation_size2=None,
        cls_head=False,
        Linear_nums=1
    ):
        super(OutputLayer, self).__init__()

        self.num_classes = num_classes
        modules = []
        if representation_size1:
            modules.append(nn.Linear(embedding_dim, representation_size1))
            modules.append(nn.Sigmoid())
            modules.append(nn.Linear(representation_size1, representation_size2))
            modules.append(nn.Sigmoid())
            modules.append(nn.Linear(representation_size2, 128))
            modules.append(nn.Sigmoid())
            modules.append(nn.Linear(128, num_classes))
            modules.append(nn.Sigmoid())
        else:
            if Linear_nums == 1:
                modules.append(nn.Linear(embedding_dim, num_classes))
            elif Linear_nums == 2:
                modules.append(nn.Linear(embedding_dim, 128))
                modules.append(nn.Sigmoid())
                modules.append(nn.Linear(128, num_classes))
            elif Linear_nums == 3:
                modules.append(nn.Linear(embedding_dim, 150))
                modules.append(nn.Sigmoid())
                modules.append(nn.Linear(150, 100))
                modules.append(nn.Sigmoid())
                modules.append(nn.Linear(100, num_classes))
        self.net = nn.Sequential(*modules)

        if cls_head:
            self.to_cls_token = nn.Identity()

        self.cls_head = cls_head
        self.num_classes = num_classes
        self._init_weights()

    def _init_weights(self):
        for name, module in self.net.named_children():
            if isinstance(module, nn.Linear):
                if module.weight.shape[0] == self.num_classes:
                    nn.init.normal_(module.weight, mean=0, std=0.1)
                    nn.init.normal_(module.bias, 0.1)

    def forward(self, x):
        if self.cls_head:
            x = self.to_cls_token(x[:, 0])
        else:
            x = torch.mean(x, dim=1)

        return self.net(x)

class CNN(nn.Module):
    def __init__(self, embedding_dim, opt, mlp_ratio, qkv_bias=True,use_revised_ffn=False,
        dropout_rate=0.7,attn_dropout_rate=0.7):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=opt.Kernel_size1,  stride=opt.Stride1).cuda()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2).cuda()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2).cuda()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=opt.Kernel_size2, stride=opt.Stride2).cuda()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        self.transformer = Transformer(
            dim=embedding_dim,
            depth=opt.num_layers,
            heads=opt.num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout_rate,
            dropout=dropout_rate,
            qkv_bias=qkv_bias,
            revised=use_revised_ffn,
        )
        self.post_transformer_ln = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(x, dim=1)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.post_transformer_ln(x)
        return x

class Submodel(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Submodel, self).__init__()
        self.n1 = nn.Linear(input_dim, 200).cuda()
        self.n2 = nn.Linear(200, output_dim).cuda()

    def forward(self, x):
        x = torch.sigmoid(self.n1(x))
        x = torch.sigmoid(self.n2(x))
        return x

class AesModel(nn.Module):
    def __init__(
        self,embedding_dim=768,opt=None):
        super(AesModel, self).__init__()
        mlp_ratio = float(opt.mlp_ratio)
        self.CNN = CNN(embedding_dim, opt, mlp_ratio)
        
        self.submodels = nn.ModuleList([Submodel(embedding_dim, 1) for _ in range(opt.num_classes)])

    def forward(self, x):
        cnn_output = self.CNN(x)
        output = torch.cat([model(cnn_output) for model in self.submodels], dim=1)
        return output


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))).cuda()
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1))).cuda()
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)