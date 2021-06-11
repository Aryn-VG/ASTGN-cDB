import torch
import torch.nn as nn
import dgl


class ATTN(nn.Module):
    def __init__(self,args,time_encoder):
        super().__init__()
        self.n_layers = args.n_layers
        self.h_dimension = args.emb_dimension
        self.attnlayers = nn.ModuleList()
        self.mergelayers=nn.ModuleList()
        self.edge_feat_dim=args.edge_feat_dim
        self.n_heads = args.n_heads
        self.time_dim = args.time_dimension
        self.node_feat_dim = args.node_feat_dim
        self.dropout = args.dropout
        self.args=args
        self.time_encoder=time_encoder
        self.device='cuda:0' if torch.cuda.is_available() else 'cpu'

        self.query_dim = self.node_feat_dim + self.time_dim
        self.key_dim = self.node_feat_dim + self.time_dim + self.edge_feat_dim
        for i in range(0, self.n_layers):
            self.attnlayers.append(nn.MultiheadAttention(embed_dim=self.query_dim,
                                                       kdim=self.key_dim,
                                                       vdim=self.key_dim,
                                                       num_heads=self.n_heads,
                                                       dropout=self.dropout).to(self.device))
            self.mergelayers.append(MergeLayer(self.query_dim, self.node_feat_dim, self.node_feat_dim, self.h_dimension).to(self.device))

    def C_compute(self,edges):
        te_C = self.time_encoder(edges.data['timestamp'] - edges.src['last_update'])
        if self.args.bandit == 'OriBandit':
            eid=edges.data[dgl.EID].view(-1,1).float()
            q_ij=edges.data['q_ij'].view(-1,1).float()
            C = torch.cat([edges.src['emb'], edges.data['feat'], te_C,eid,q_ij], dim=1)
        elif self.args.bandit=='cDyBandit':
            eid = edges.data[dgl.EID].view(-1, 1).float()
            C = torch.cat([edges.src['emb'], edges.data['feat'], te_C, eid], dim=1)
        else:
            C = torch.cat([edges.src['emb'], edges.data['feat'], te_C], dim=1)
        #print(C.size())
        return {'C': C}

    def h_compute(self,nodes):

        if self.args.bandit=='OriBandit':
            C = nodes.mailbox['C'][:, :, :-2]
            eid=nodes.mailbox['C'][:,:,-2].view(-1).detach().long()
            q_ij=nodes.mailbox['C'][:,:,-1].view(-1).detach()
        elif self.args.bandit=='cDyBandit':
            C = nodes.mailbox['C'][:, :, :-1]
            eid = nodes.mailbox['C'][:, :, -1].view(-1).detach().long()
        else:
            C = nodes.mailbox['C']
        #print(q_ij,eid)
        C=C.permute([1,0,2])#convert to [num_node,num_neighbor,feat_dim]
        key = C.to(self.device)

        te_q=self.time_encoder(torch.zeros(nodes.batch_size()).to(self.device))
        query = torch.cat([nodes.data['emb'], te_q],dim=1).unsqueeze(dim=0)

        h_before,att= self.attnlayers[self.l](query=query, key=key, value=key)
        if self.args.bandit == 'OriBandit':
            qij_sum=torch.sum(q_ij)
            att_sum=torch.sum(att).view(-1)
            att=qij_sum*att/att_sum
            if self.l==self.n_layers-1:
                self.attn_map[eid]=att.view(-1)
        elif self.args.bandit=='cDyBandit':
            att_sum = torch.sum(att).view(-1)
            att = att / att_sum
            if self.l==self.n_layers-1:
                self.attn_map[eid]=att.view(-1)


        #print(self.attn_map)
        h_before=h_before.squeeze(0)

        h= self.mergelayers[self.l](nodes.data['emb'], h_before)
        return {'emb':h}

    def forward(self, blocks):
        for l in range(self.n_layers):
            self.l = l
            if self.args.bandit=='OriBandit' and self.l==self.n_layers-1:
                attn_map=torch.ones((blocks[self.l].number_of_edges())).to(self.device)
                self.attn_map=-attn_map
            elif self.args.bandit=='cDyBandit' and self.l==self.n_layers-1:
                attn_map = torch.ones((blocks[self.l].number_of_edges())).to(self.device)
                self.attn_map = -attn_map
            else:
                self.attn_map=None
            blocks[l].update_all(self.C_compute,self.h_compute)
            if l!=self.n_layers-1:
                blocks[l+1].srcdata['emb']=blocks[l].dstdata['emb']

        return blocks,self.attn_map

class MergeLayer(torch.nn.Module):
    '''(dim1+dim2)->dim3->dim4'''

    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)

