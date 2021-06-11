import dgl
import torch
import dgl.function as fn

class TemporalEdgeCollator(dgl.dataloading.EdgeCollator):
    def __init__(self,args,g,eids,block_sampler,g_sampling=None,exclude=None,
                reverse_eids=None,reverse_etypes=None,negative_sampler=None):
        super(TemporalEdgeCollator,self).__init__(g,eids,block_sampler,
                                                 g_sampling,exclude,reverse_eids,reverse_etypes,negative_sampler)
        self.args=args
    def collate(self,items):
        current_ts=self.g.edata['timestamp'][items[0]]     #only sample edges before current timestamp
        self.block_sampler.ts=current_ts
        neg_pair_graph=None
        if self.negative_sampler is None:
            input_nodes,pair_graph,blocks=self._collate(items)
        else:
            input_nodes,pair_graph,neg_pair_graph,blocks=self._collate_with_negative_sampling(items)
        if self.args.n_layers>1:
            self.block_sampler.frontiers[0].add_edges(*self.block_sampler.frontiers[1].edges())
        frontier=dgl.reverse(self.block_sampler.frontiers[0])
        return input_nodes, pair_graph, neg_pair_graph, blocks, frontier, current_ts


class EvalSampler(dgl.dataloading.BlockSampler):

    def __init__(self, args,replace=False, return_eids=False):
        super().__init__(args.n_layers,return_eids)

        # if args.fanouts:
        #     assert len(args.fanouts) != args.n_layers, 'fanouts should match the n_layers'
        #     self.fanouts = args.fanouts
        # elif args.n_degrees != 0:
        #     self.fanouts = [args.n_degrees for _ in range(args.n_layers)]
        # else:
        #     self.fanouts = None
        self.replace = replace
        self.ts = 0
        self.args = args
        self.frontiers = [None for _ in range(args.n_layers)]

    def sample_frontier(self, block_id, g, seed_nodes):
        # List of neighbors to sample per edge type for each GNN layer, starting from the first layer.
        g = dgl.in_subgraph(g, seed_nodes)
        g.remove_edges(torch.where(g.edata['timestamp'] > self.ts)[0])
        frontier=g
        self.frontiers[block_id] = frontier
        return frontier

class TrainSampler(dgl.dataloading.BlockSampler):

    def __init__(self, args, replace=False, return_eids=False):
        super().__init__(args.n_layers, return_eids)

        if args.fanouts:
            assert len(args.fanouts) != args.n_layers, 'fanouts should match the n_layers'
            self.fanouts = args.fanouts
        elif args.n_degrees!=0:
            self.fanouts = [args.n_degrees for _ in range(args.n_layers)]
        else:
            self.fanouts=None

        self.replace = replace
        self.ts = 0
        self.args = args
        self.frontiers = [None for _ in range(args.n_layers)]
        self.sigmoid=torch.nn.Sigmoid()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def sample_timespan(self, edges):
        timespan = edges.dst['sample_time'] - edges.data['timestamp']

        return {'timespan': timespan}

    def sample_time(self, edges):
        return {'st': edges.data['timestamp']}

    def prob_broadcast(self,edges):
        prob=edges.src['prob']
        return {'prob': prob}

    def update_timespan(self,edges):
        update_span=edges.data['timestamp'] - edges.src['last_update']
        return {'update_span':update_span}

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id] if self.fanouts is not None else None

        # List of neighbors to sample per edge type for each GNN layer, starting from the first layer.
        g = dgl.in_subgraph(g, seed_nodes)
        g.remove_edges(torch.where(g.edata['timestamp'] > self.ts)[0])
        if self.args.causual_sampling:
            if block_id != self.args.n_layers - 1:
                g.dstdata['sample_time'] = self.frontiers[block_id + 1].srcdata['sample_time']
                g.apply_edges(self.sample_timespan)
                g.remove_edges(torch.where(g.edata['timespan'] < 0)[0])
            g_re=dgl.reverse(g,copy_edata=True,copy_ndata=True)
            g_re.update_all(self.sample_time,fn.max('st','sample_time'))
            g=dgl.reverse(g_re,copy_edata=True,copy_ndata=True)

        if fanout is None:
            frontier = g
        else:
            if block_id == self.args.n_layers - 1:
                if self.args.bandit=='OriBandit':
                    frontier = dgl.sampling.sample_neighbors(g,seed_nodes,fanout,prob='q_ij')
                elif self.args.bandit=='cDyBandit':#remained to be done
                    g.apply_edges(self.update_timespan)
                    time_re=torch.div(1.,g.edata['update_span']+1)#.to(self.device)
                    theta_cum = torch.bmm(g.ndata['A_cum'].inverse(),
                                          g.ndata['b_cum'].unsqueeze(dim=2))  # [bs,emb_dim,1]=[bs,emb_dim,emb_dim]*[bs,emb_dim,1]
                    reward_term = torch.bmm(g.ndata['emb'].unsqueeze(dim=1), theta_cum).view(-1)  # 每个邻居的采样概率的reward部分

                    #吧theta_cum更新到图上
                    g.ndata['theta_cum']=theta_cum

                    exploration_term = torch.bmm(g.ndata['emb'].unsqueeze(dim=1), g.ndata['A_cum'].inverse())
                    exploration_term = torch.bmm(exploration_term, g.ndata['emb'].unsqueeze(dim=2)).view(-1)  # [bs,]
                    g.ndata['prob'] = reward_term + self.args.alpha * torch.sqrt(exploration_term) # [bs,]
                    g.apply_edges(self.prob_broadcast)
                    g.edata['prob']=torch.div(g.edata['prob'],torch.sum(g.edata['prob']))
                    g.edata['prob']+=time_re
                    frontier = dgl.sampling.select_topk(g, fanout, 'prob', seed_nodes)
                else:
                    frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)

            else:
                frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)

        self.frontiers[block_id] = frontier
        return frontier


def dataloader(args,g):
    origin_num_edges = g.num_edges() // 2

    train_eid = torch.arange(0, int(0.7 * origin_num_edges))
    val_eid = torch.arange(int(0.7 * origin_num_edges), int(0.85 * origin_num_edges))
    test_eid = torch.arange(int(0.85 * origin_num_edges), origin_num_edges)
    exclude, reverse_eids = None, None

    negative_sampler = dgl.dataloading.negative_sampler.Uniform(1)
    train_sampler = TrainSampler(args,  return_eids=False)
    val_sampler=EvalSampler(args, return_eids=False)
    train_collator = TemporalEdgeCollator(args,g, train_eid, train_sampler, exclude=exclude, reverse_eids=reverse_eids,
                                          negative_sampler=negative_sampler)

    train_loader = torch.utils.data.DataLoader(
        train_collator.dataset, collate_fn=train_collator.collate,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers)
    val_collator = TemporalEdgeCollator(args,g, val_eid, val_sampler, exclude=exclude, reverse_eids=reverse_eids,
                                        negative_sampler=negative_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_collator.dataset, collate_fn=val_collator.collate,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers)
    test_collator = TemporalEdgeCollator(args,g, test_eid, val_sampler, exclude=exclude, reverse_eids=reverse_eids,
                                         negative_sampler=negative_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_collator.dataset, collate_fn=test_collator.collate,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers)
    return train_loader, val_loader, test_loader, val_eid.shape[0], test_eid.shape[0]