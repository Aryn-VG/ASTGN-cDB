import dgl.function as fn
import torch
class OriBandit_sampler:
    def __init__(self,eta,T,k):
        self.eta=eta
        self.T=T
        self.k=k
        self.device='cuda:0' if torch.cuda.is_available() else 'cpu'
        self.qij_soft=torch.nn.Softmax(dim=0)

    def init_indegree(self,edges):
        in_degree=edges.dst['indegree']
        return{'in_degree':in_degree}

    def weight_update(self,blocks,att_map):
        q_ij = blocks[-1].edata['q_ij']
        weight =blocks[-1].edata['weight']

        blocks[-1].dstdata['indegree']=blocks[-1].in_degrees()
        blocks[-1].apply_edges(self.init_indegree)


        self.n = blocks[-1].edata['in_degree'].float()
        n_min=torch.ones_like(self.n).float().to(self.device)
        self.n=torch.max(n_min,self.n)
        self.k_v = self.k * torch.ones_like(self.n)
        self.k_v = torch.min(self.k_v, self.n)

        r_ij = torch.div(torch.mul(att_map,att_map),self.k_v)
        r_ij = torch.div(r_ij,torch.mul(q_ij, q_ij))
        r_ij_hat= torch.div(r_ij,q_ij)
        delta=torch.sqrt((1-self.eta)*self.eta**4*self.k_v**5*torch.log(self.n/self.k_v)/(self.T*self.n**4))
        r_ij=torch.exp(torch.mul(r_ij_hat,delta))/self.n
        max_reward=2 * torch.ones_like(r_ij)
        self.weight= torch.mul(weight,torch.min(r_ij,max_reward))
        min_weight=torch.Tensor([1e-6]).to(self.device)
        self.weight=torch.max(self.weight,min_weight)
        blocks[-1].edata['weight']=self.weight
        return blocks


    def weight_sum(self,edges):
        return {"w_s":edges.data['weight']}
    def weight_sum_2edges(self,edges):
        return{'weight_sum':edges.dst['weight_sum']}

    def qij_sum(self,edges):
        return{'q_s':edges.data['q_ij']}

    def qij_sum_2edges(self,edges):
        return {'qij_sum': edges.dst['qij_sum']}


    def prob_update(self,blocks):
        blocks[-1].update_all(self.weight_sum,fn.sum('w_s','weight_sum'))
        blocks[-1].apply_edges(self.weight_sum_2edges)
        weight_sum=blocks[-1].edata['weight_sum']

        q_ij = torch.div(self.weight, weight_sum) * (1 - self.eta) + self.eta /self.n
        q_ij = torch.abs(q_ij)
        blocks[-1].edata['q_ij'] = q_ij

        blocks[-1].update_all(self.qij_sum, fn.sum('q_s', 'qij_sum'))
        blocks[-1].apply_edges(self.qij_sum_2edges)
        q_sum=blocks[-1].edata['qij_sum']
        q_ij=q_ij/q_sum

        min_qij = torch.Tensor([1e-6]).to(self.device)
        q_ij = torch.max(q_ij, min_qij)
        blocks[-1].edata['q_ij'] = q_ij
        return blocks

