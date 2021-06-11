from pytorch_lightning.metrics.functional import accuracy, auroc, average_precision
import torch
import numpy as np
import dgl
import time
import dgl.function as fn

def get_current_ts( pos_graph, neg_graph):
    with pos_graph.local_scope():
        pos_graph_ = dgl.add_reverse_edges(pos_graph, copy_edata=True)
        pos_graph_.update_all(fn.copy_e('timestamp', 'times'), fn.max('times','ts'))
        current_ts = pos_ts = pos_graph_.ndata['ts']
        num_pos_nodes = pos_graph_.num_nodes()
    with neg_graph.local_scope():
        neg_graph_ = dgl.add_reverse_edges(neg_graph)
        neg_graph_.edata['timestamp'] = pos_graph_.edata['timestamp']
        neg_graph_.update_all(fn.copy_e('timestamp', 'times'), fn.max('times','ts'))
        num_pos_nodes = torch.where(pos_graph_.ndata['ts']>0)[0].shape[0]
        pos_ts = pos_graph_.ndata['ts'][:num_pos_nodes]
        neg_ts = neg_graph_.ndata['ts'][num_pos_nodes:]
        current_ts = torch.cat([pos_ts,neg_ts])
    return current_ts, pos_ts, num_pos_nodes
def eval_epoch(args,g, dataloader, attn,decoder,loss_fcn, device,num_samples):
    m_ap, m_auc, m_acc = [[], [], []] if 'LP' in args.tasks else [0,0,0]
    labels_all = torch.zeros((num_samples))
    logits_all = torch.zeros((num_samples))
    m_loss = []
    m_infer_time = []
    with torch.no_grad():
        attn.eval()
        decoder.eval()
        #loss = torch.tensor(0)
        for batch_idx, (input_nodes, pos_graph, neg_graph, blocks, frontier, current_ts) in enumerate(dataloader):


            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            n_sample = pos_graph.num_edges()
            start_idx = batch_idx * n_sample
            end_idx = min(num_samples, start_idx + n_sample)
            for j in range(args.n_layers):
                blocks[j] = blocks[j].to(device)

            current_ts, pos_ts, num_pos_nodes = get_current_ts(pos_graph, neg_graph)
            pos_graph.ndata['ts'] = current_ts


            start = time.time()
            blocks,att_map=attn.forward(blocks)
            emb = blocks[-1].dstdata['emb']

            # blocks = bandi_sampler.weight_update(blocks, att_map)
            # blocks = bandi_sampler.prob_update(blocks)

            logits, labels = decoder(emb, pos_graph, neg_graph)
            end = time.time() - start
            m_infer_time.append(end)

            loss = loss_fcn(logits, labels)
            m_loss.append(loss.item())

            g.ndata['last_update'][pos_graph.ndata[dgl.NID][:num_pos_nodes]] = pos_ts#.to('cpu')
            # g.edata['q_ij'][blocks[-1].edata['eid']] = blocks[-1].edata['q_ij'].cpu()
            # g.edata['weight'][blocks[-1].edata['eid']] = blocks[-1].edata['weight'].cpu()
            #g.ndata['emb'][pos_graph.ndata[dgl.NID]] = emb.to('cpu')
            if 'LP' in args.tasks:
                labels=labels.int()
                pred = logits.sigmoid() > 0.5
                m_ap.append(average_precision(logits, labels).cpu().numpy())
                m_auc.append(auroc(logits, labels).cpu().numpy())
                m_acc.append(accuracy(pred, labels).cpu().numpy())
            else:
                labels_all[start_idx:end_idx] = labels
                logits_all[start_idx:end_idx] = logits
    if 'LP' in args.tasks:
        ap, auc, acc = np.mean(m_ap), np.mean(m_auc), np.mean(m_acc)
    else:
        pred_all = logits_all.sigmoid() > 0.5
        ap = average_precision(logits_all, labels_all).cpu().item()
        auc = auroc(logits_all, labels_all).cpu().item()
        acc = accuracy(pred_all, labels_all).cpu().item()

    attn.train()
    decoder.train()
    return ap, auc, acc, np.mean(m_loss),np.sum(m_infer_time)