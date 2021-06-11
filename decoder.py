import torch
from torch import nn
import dgl.function as fn



class Decoder(nn.Module):
    def __init__(self, args, nfeat_dim):
        super().__init__()
        self.args = args
        dropout = args.dropout
        if 'LP' in args.tasks:
            self.linkpredlayer = LinkPredLayer(nfeat_dim * 2, 1, dropout)
        if 'NC' in args.tasks:
            self.nodeclaslayer = NodeClasLayer(nfeat_dim, 1, dropout)

    def linkpred(self, block_outputs, pos_graph, neg_graph):
        with neg_graph.local_scope():
            neg_graph.ndata['feat'] = block_outputs
            neg_graph.apply_edges(linkpred_concat)
            neg_emb = neg_graph.edata['emb']
        with pos_graph.local_scope():
            pos_graph.ndata['feat'] = block_outputs
            pos_graph.apply_edges(linkpred_concat)
            pos_emb = pos_graph.edata['emb']
        logits = self.linkpredlayer(torch.cat([pos_emb, neg_emb]))
        labels = torch.zeros_like(logits)
        labels[:pos_emb.shape[0]] = 1
        return logits, labels

    def nodeclas(self, block_outputs, pos_graph, fraud_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['feat'] = block_outputs
            pos_graph.apply_edges(fn.copy_u('feat', 'emb'))
            node_emb = pos_graph.edata['emb']
        labels = pos_graph.edata['label']
        logits = self.nodeclaslayer(node_emb)
        return logits, labels

    def forward(self, block_outputs, pos_graph, neg_graph):
        if 'LP' in self.args.tasks:
            logits, labels = self.linkpred(block_outputs, pos_graph, neg_graph)
        elif 'NC' in self.args.tasks:
            logits, labels = self.nodeclas(block_outputs, pos_graph, neg_graph)
        else:
            logits, labels=None,None
        return logits.squeeze(), labels.squeeze()

def linkpred_concat(edges):
    return {'emb': torch.cat([edges.src['feat'], edges.dst['feat']], 1)}

class LinkPredLayer(nn.Module):
    def __init__(self, in_dim, class_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        hidden_dim = in_dim // 2
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, class_dim)
        self.act = nn.ReLU()
        self.act2=nn.Sigmoid()

        self.dropout = nn.Dropout(p=dropout, inplace=True)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, h):
        h = self.dropout(h)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.act2(h)
        return h


class NodeClasLayer(nn.Module):
    def __init__(self, in_dim, class_dim, dropout=0.1):
        super().__init__()
        hidden_dim = in_dim//2
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.out = nn.Linear(hidden_dim//2, class_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout, inplace=True)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, h):
        h = self.dropout(h)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.out(h)

        return h
