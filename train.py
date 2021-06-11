from args import get_args
from dataloader import dataloader
from dgl.data.utils import load_graphs
import dgl.function as fn
import torch
import dgl
from embeding import ATTN
from time_encode import TimeEncode
from decoder import  Decoder
from val_eval import get_current_ts,eval_epoch
from bandit_sampler import OriBandit_sampler
import logging
import torch.nn as nn
def get_log(file_name):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(file_name, mode='a')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

if __name__ == '__main__':
    args = get_args()
    logger = get_log('log.txt')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    #the path of data
    if args.dataset=='mooc':
        g = load_graphs('mooc.bin')[0][0]
        g = dgl.add_reverse_edges(g, copy_edata=True)
        g.edata['timestamp'] = g.edata['timestamp'].float()
        g.edata['feat'] = g.edata['feats']
        args.edge_feat_dim=4

    else:
        g = load_graphs(args.dataset + '.dgl')[0][0]


    efeat_dim= g.edata['feat'].shape[1]
    if efeat_dim!=args.edge_feat_dim:
        g.edata['feat']=torch.zeros((g.number_of_edges(), args.edge_feat_dim))



    if args.dataset in ['wiki','reddit']:
        node_feature = torch.zeros((g.number_of_nodes(), args.node_feat_dim))

        g.ndata['feat'] = node_feature
    if not efeat_dim or efeat_dim==0:
        g.edata['feat']=torch.zeros((g.number_of_edges(), args.edge_feat_dim))



    #initialize
    mat_norm = nn.LayerNorm([args.emb_dimension,args.emb_dimension], elementwise_affine=False).to(device)
    vec_norm = nn.LayerNorm([args.emb_dimension], elementwise_affine=False).to(device)
    time_encoder = TimeEncode(args.time_dimension).to(device)
    emb_updater = ATTN(args, time_encoder).to(device)
    decoder=Decoder(args,args.emb_dimension).to(device)
    loss_fcn = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(list(decoder.parameters())+list(emb_updater.parameters()), lr=args.lr)

    if args.bandit=="OriBandit":
        k = torch.tensor(args.n_degrees).float().to(device)
        eta = torch.tensor(args.eta).float().to(device)
        T = torch.tensor(args.T).float().to(device)
        bandit_sampler=OriBandit_sampler(eta,T,k)
        # initialize q_ij&weight
        g.ndata['indegree'] = g.in_degrees()
        def init_indegree(edges):
            in_degree = edges.dst['indegree']
            return {'in_degree': in_degree}
        def init_qij(edges):
            prob = torch.div(1.0, edges.dst['indegree'])
            return {'q_ij': prob}
        def init_weight(g):
            weight = torch.ones((g.number_of_edges()))
            g.edata['weight'] = weight
            return g
        g.apply_edges(init_indegree)
        g.edata['timestamp'] += torch.tensor(1)
        eid_of_g = torch.linspace(0, g.number_of_edges() - 1, g.number_of_edges()).long()

        g.edata['eid'] = eid_of_g
    elif args.bandit=="cDyBandit":
        def cDyBandit_init(g):
            A = torch.eye(args.emb_dimension).expand(g.num_nodes(), args.emb_dimension, args.emb_dimension)# [bs,emb_size,emb_size]
            b = torch.zeros((args.emb_dimension)).expand(g.num_src_nodes(), args.emb_dimension) # [bs,emb_size]
            g.srcdata['A_pre'], g.srcdata['b_pre'] = A, b
            g.srcdata['A_cum'], g.srcdata['b_cum'] = A, b
            g.srcdata['A_cur'], g.srcdata['b_cur'] = A, b
            return g

        def change_detecting(g):
            theta_pre = torch.bmm(g.srcdata['A_pre'].inverse(),
                                  g.srcdata['b_pre'].unsqueeze(dim=2))  # [bs,d,1]=[bs,d,d]*[bs,d,1]
            reward_pred = torch.bmm(g.srcdata['emb'].unsqueeze(dim=1), theta_pre).view(-1)  # 现在节点的emb和上次的估计theta_pre

            reward_gap = reward_pred - g.srcdata['pre_reward']
            change_point_index = torch.abs(reward_gap) >=args.delta
            unchange_point_index = ~change_point_index
            # print("reward_pred:max(%f),mean(%f);pre_reward:max(%f),mean(%f)"%(torch.max(reward_pred),torch.mean(reward_pred),torch.max(g.srcdata['pre_reward']),torch.mean(g.srcdata['pre_reward'])))

            # changed part
            if torch.sum(change_point_index):
                g.srcdata['A_pre'][change_point_index], g.srcdata['b_pre'][change_point_index] = g.srcdata['A_cur'][change_point_index], g.srcdata['b_cur'][change_point_index]
                g.srcdata['A_cum'][change_point_index], g.srcdata['b_cum'][change_point_index] = g.srcdata['A_cur'][change_point_index], g.srcdata['b_cur'][change_point_index]

                g.srcdata['A_cur'][change_point_index] = torch.eye(args.emb_dimension).expand(torch.sum(change_point_index), args.emb_dimension, args.emb_dimension)  # [bs,d,d]
                g.srcdata['b_cur'][change_point_index] = torch.zeros((args.emb_dimension)).expand(torch.sum(change_point_index), args.emb_dimension)  # [bs,d]


            # unchanged part
            g.srcdata['A_pre'][unchange_point_index] = g.srcdata['A_pre'][unchange_point_index] + torch.bmm(g.srcdata['emb'][unchange_point_index].unsqueeze(dim=1),
                                                            g.srcdata['emb'][unchange_point_index].unsqueeze(dim=2))
            g.srcdata['b_pre'][unchange_point_index] = g.srcdata['b_pre'][unchange_point_index] + torch.matmul(g.srcdata['pre_reward'][unchange_point_index], g.srcdata['emb'][unchange_point_index])
            g.srcdata['A_cur'][unchange_point_index] = g.srcdata['A_cur'][unchange_point_index] - torch.bmm(g.srcdata['emb'][unchange_point_index].unsqueeze(dim=1),
                                                            g.srcdata['emb'][unchange_point_index].unsqueeze(dim=2))
            g.srcdata['b_cur'][unchange_point_index] = g.srcdata['b_cur'][unchange_point_index] - torch.matmul(g.srcdata['pre_reward'][unchange_point_index], g.srcdata['emb'][unchange_point_index])
            return g

        # def reward_computing(g):
        #     # Reward computing
        #     g.srcdata['pre_reward'] = torch.bmm(g.srcdata['emb'].unsqueeze(dim=1),
        #                                            g.srcdata['theta_cum']).view(-1)
        #     return g

        def Ab_updating(g):
            # Updating A&b
            g.srcdata['A_cur'] = g.srcdata['A_cur'] + torch.bmm(g.srcdata['emb'].unsqueeze(dim=1),
                                                            g.srcdata['emb'].unsqueeze(dim=2))
            g.srcdata['A_cum'] = g.srcdata['A_cum'] + torch.bmm(g.srcdata['emb'].unsqueeze(dim=1),
                                                            g.srcdata['emb'].unsqueeze(dim=2))

            g.srcdata['b_cum'] = g.srcdata['b_cum'] + torch.matmul(g.srcdata['reward'], g.srcdata['emb'])
            g.srcdata['b_cur'] = g.srcdata['b_cur'] + torch.matmul(g.srcdata['reward'], g.srcdata['emb'])
            return g


        bandit_sampler = None

    else:
        bandit_sampler = None
        pass
    train_loader, val_loader, test_loader, val_num, test_enum = dataloader(args, g)

    def epoch_graph_init(g):
        g.ndata['emb']=torch.rand((g.number_of_nodes(), args.emb_dimension))
        g.ndata['last_update'] = torch.zeros(g.number_of_nodes())
        g.ndata['pre_reward']=torch.zeros((g.number_of_nodes(),))
        if args.bandit=='OriBandit':
            g.apply_edges(init_qij)
            init_weight(g)
        elif args.bandit=='cDyBandit':
            g=cDyBandit_init(g)
        else:pass
        return g


    #training
    for i in range(args.n_epochs):
        g=epoch_graph_init(g)
        decoder.train()
        time_encoder.train()
        emb_updater.train()
        g = g.to(device)

        for batch_id, (input_nodes, pos_graph, neg_graph, blocks, frontier, current_ts) in enumerate(train_loader):
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            for j in range(args.n_layers):
                blocks[j] = blocks[j].to(device)

            current_ts, pos_ts, num_pos_nodes = get_current_ts(pos_graph, neg_graph)
            pos_graph.ndata['ts'] = current_ts

            # if args.bandit=='cDyBandit':
            #     # blocks[-1]=reward_computing(blocks[-1])
            #     # blocks[-1]=Ab_updating(blocks[-1])
            #     # b_cum作为emb的初始化
            #     blockP:s[-1].srcdata['emb']+=blocks[-1].srcdata['b_cum']
            #     blocks[-1].dstdata['emb']+=blocks[-1].dstdata['b_cum']
            # else:
            #     pass

            blocks,att_map=emb_updater.forward(blocks)

            emb=blocks[-1].dstdata['emb']

            if batch_id!=0:
                # backward
                logits, labels = decoder(emb, pos_graph, neg_graph)
                # pred = logits.sigmoid() > 0.5
                # ap = average_precision(logits, labels)
                loss = loss_fcn(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if batch_id%1000==1:
                #     logger.info('batch:%d,loss:%f'%(batch_id,loss.item()))

            if args.bandit=='OriBandit':
                # update bandit sampler
                blocks=bandit_sampler.weight_update(blocks,att_map)
                blocks=bandit_sampler.prob_update(blocks)
            elif args.bandit == 'cDyBandit':
                # reward compute
                def attn_message(edges):
                    return {'att':edges.data['attn']}

                blocks[-1].edata['attn']=att_map
                #sum attn to srcnode
                block_g=dgl.block_to_graph(blocks[-1])
                reverse_block_g=dgl.reverse(block_g,copy_ndata=True,copy_edata=True)
                reverse_block_g.update_all(attn_message,fn.sum('att','attn_sum'))
                reward_cur=reverse_block_g.dstdata['attn_sum']
                blocks[-1].srcdata['reward']=reward_cur
                blocks[-1] = Ab_updating(blocks[-1])

                #Change detecting
                blocks[-1]=change_detecting(blocks[-1])

                #pre_reward updating
                blocks[-1].srcdata['pre_reward'] = reward_cur
            else:
                pass

            with torch.no_grad():
                g.ndata['last_update'][pos_graph.ndata[dgl.NID][:num_pos_nodes]] = pos_ts#.cpu()
                if args.bandit == 'OriBandit':
                    g.edata['q_ij'][blocks[-1].edata['eid']] = blocks[-1].edata['q_ij'].cpu()
                    g.edata['weight'][blocks[-1].edata['eid']] = blocks[-1].edata['weight'].cpu()
                elif args.bandit=='cDyBandit':
                    for l in range(args.n_layers):
                        idx_src_block = blocks[l].srcdata['_ID']
                        g.ndata['A_cum'][idx_src_block] = mat_norm(blocks[l].srcdata['A_cum'])
                        g.ndata['A_cur'][idx_src_block] = mat_norm(blocks[l].srcdata['A_cur'])
                        g.ndata['A_pre'][idx_src_block] = mat_norm(blocks[l].srcdata['A_pre'])

                        g.ndata['b_cum'][idx_src_block] = vec_norm(blocks[l].srcdata['b_cum'])
                        g.ndata['b_cur'][idx_src_block] =vec_norm(blocks[l].srcdata['b_cur'])
                        g.ndata['b_pre'][idx_src_block] = vec_norm(blocks[l].srcdata['b_pre'])
                    idx_src_block = blocks[-1].srcdata['_ID']
                    idx_dst_block = blocks[-1].dstdata['_ID']
                    # print(g.device,g.ndata['emb'].device)
                    g.ndata['pre_reward'][idx_src_block] = blocks[-1].srcdata['pre_reward']
                    # g.ndata['emb'][idx_src_block] = blocks[-1].srcdata['emb']
                    g.ndata['emb'][idx_dst_block] = blocks[-1].dstdata['emb']
                else:
                    pass

                # g.ndata['emb'][pos_graph.ndata[dgl.NID]] = emb.to('cpu')
        val_ap, val_auc, val_acc, val_loss ,time_c= eval_epoch(args,g, val_loader, emb_updater, decoder,loss_fcn, device,val_num)
        g = g.cpu()
        # print("epoch:%d,loss:%f,ap:%f,time_consume:%f" % (i, val_loss, val_ap,time_c))
        logger.info("epoch:%d,loss:%f,ap:%f,time_consume:%f" % (i, val_loss, val_ap,time_c))

    test_ap, test_auc, test_acc, test_loss, test_time = eval_epoch(args, g, test_loader, emb_updater, decoder,
                                                                   loss_fcn, device, test_enum)
    logger.info("Test:loss:%f,ap:%f,auc:%f,acc:%f,time:%f" % (test_loss, test_ap, test_auc, test_acc, test_time))
