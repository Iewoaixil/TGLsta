import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
from torch.utils.data import DataLoader
from sklearn import preprocessing
import numpy as np
import argparse
import torch
from random import sample
import random
import math
import time
from model import CLIP, tokenize
from data import DataHelper
from sklearn import preprocessing

import math
from torch import nn
import dgl
from utils import batcher
from graph_encoder import GraphEncoder
from utils import _rwr_trace_to_dgl_graph
import warnings
from collections import defaultdict, namedtuple
warnings.filterwarnings("ignore")

class MemoryMoCo(nn.Module):
    
    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer("params", torch.tensor([-1]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer(
            "memory", torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv)
        )
        print("using queue shape: ({},{})".format(self.queueSize, inputSize))
    def forward(self, q, k):
        batchSize = q.shape[0]
        k = k.detach()
        Z = self.params[0].item()
        
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)
        out = torch.cat((l_pos, l_neg), dim=1)
        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0:
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            
            out = torch.div(out, Z).squeeze().contiguous()
        
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out

def generate_embeddings(graphs, idx, step_dist=[1.0, 0.0, 0.0], rw_hops=32, restart_prob=0.8, positional_embedding_size=32):
    node_idx = idx
    step = np.random.choice(len(step_dist), 1, p=step_dist)[0]
    if step == 0:   
        other_node_idx = node_idx
    else:
        other_node_idx = dgl.contrib.sampling.random_walk(
            g=graphs, seeds=[node_idx], num_traces=1, num_hops=step
        )[0][0][-1].item()

    max_nodes_per_seed = max(
        rw_hops,
        int(
            (
                    graphs.out_degree(node_idx)
                    * math.e
                    / (math.e - 1)
                    / restart_prob
            )
            + 0.5
        ),
    )

    traces = dgl.contrib.sampling.random_walk_with_restart(
        graphs,
        seeds=[node_idx, other_node_idx],
        restart_prob=restart_prob,
        max_nodes_per_seed=max_nodes_per_seed,
    )

    graph_q = _rwr_trace_to_dgl_graph(
        g=graphs,
        seed=node_idx,
        trace=traces[0],
        positional_embedding_size=positional_embedding_size,
    )
    graph_k = _rwr_trace_to_dgl_graph(
        g=graphs,
        seed=other_node_idx,
        trace=traces[1],
        positional_embedding_size=positional_embedding_size,
    )
    return graph_q, graph_k    

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
class NCESoftmaxLoss(nn.Module):
    

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss
    
class NCESoftmaxLossNS(nn.Module):
    

    def __init__(self):
        super(NCESoftmaxLossNS, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        
        label = torch.arange(bsz).cuda().long()
        loss = self.criterion(x, label)
        return loss
    
def moment_update(model, model_ema, m):
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)
    
def main(args):
    setup_seed(seed)
    model = CLIP(args).to(device)
    Data = DataHelper(arr_edge_index, args)
    model.train()
    gcc_model, gcc_model_ema = [
        GraphEncoder(
            positional_embedding_size=args.positional_embedding_size,
            max_node_freq=args.max_node_freq,
            max_edge_freq=args.max_edge_freq,
            max_degree=args.max_degree,
            freq_embedding_size=args.freq_embedding_size,
            degree_embedding_size=args.degree_embedding_size,
            output_dim=args.hidden_size,
            node_hidden_dim=args.hidden_size,
            edge_hidden_dim=args.hidden_size,
            num_layers=args.num_layer,
            num_step_set2set=args.set2set_iter,
            num_layer_set2set=args.set2set_lstm_layer,
            gnn_model=args.model,
            norm=args.norm,
            degree_input=True,
        )
        for _ in range(2)
    ]
    gcc_model = gcc_model.to(device)
    gcc_model_ema = gcc_model_ema.to(device)
    gcc_model.train()
    gcc_model_ema.eval()
    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.train()

    gcc_model_ema.apply(set_bn_train)
    moment_update(gcc_model, gcc_model_ema, 0)
    contrast = MemoryMoCo(args.hidden_size, None, args.nce_k, args.nce_t, use_softmax=True).to(device)

    
    graph = dgl.DGLGraphStale()
    src, dst = arr_edge_index
    graph.add_nodes(max(src.max(), dst.max()) + 1)
    graph.add_edges(src, dst)
    graph.add_edges(dst, src)
    
    for j in range(args.epoch_num):
        loader = DataLoader(Data, batch_size=args.batch_size, shuffle=True, num_workers=10)
        for i_batch, sample_batched in enumerate(loader):
            s_n, t_n = sample_batched['s_n'], sample_batched['t_n']
            s_n_arr, t_n_arr = s_n.numpy(), t_n.numpy().reshape(-1) 
            s_n_text, t_n_text = np.array(tit_list)[s_n_arr].tolist(), np.array(tit_list)[t_n_arr].tolist()
            s_n_text, t_n_text = tokenize(s_n_text, context_length=args.context_length).to(device), tokenize(t_n_text, context_length=args.context_length).to(device)        
            graph_q_list = []
            graph_k_list = []
            for s_idx in s_n:
                graph_q, graph_k = generate_embeddings(graph, s_idx, rw_hops=args.rw_hops)
                graph_q_list.append(graph_q)
                graph_k_list.append(graph_k)
            graph_q = dgl.batch(graph_q_list)
            graph_k = dgl.batch(graph_k_list)
            graph_q = graph_q.to(device)
            graph_k = graph_k.to(device)    
            feat_q = gcc_model(graph_q)
            feat_k = gcc_model_ema(graph_k)        
            out = contrast(feat_q, feat_k)
            criterion = NCESoftmaxLoss()
            criterion = criterion.to(device)
            s_n, t_n = s_n.to(device), t_n.to(device)
            loss = model.forward(node_f, (feat_q + feat_k) / 2, edge_index, s_n, t_n, s_n_text, t_n_text, device) + criterion(out)
            if j == 0 and i_batch % 100 == 0:
                print('{}th loss in the first epoch:{}'.format(i_batch, loss))

        
        print('{}th epoch loss:{}'.format(j, loss))
    
    torch.save(model.state_dict(), './res/{}/model.pkl'.format(data_name))
    print("==> Saving...")
    state = {
        "opt": args,
        "model": gcc_model.state_dict(),
    }
    torch.save(state, './res/{}/model_gcc.pth'.format(data_name))
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--epoch_num', type=int, default=2, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--edge_coef', type=float, default=10)
    parser.add_argument('--neigh_num', type=int, default=3)

    parser.add_argument('--gnn_input', type=int, default=4096)
    parser.add_argument('--gnn_hid', type=int, default=128)
    parser.add_argument('--gnn_output', type=int, default=64)

    parser.add_argument('--context_length', type=int, default=128)

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--positional-embedding-size", type=int, default=32)
    parser.add_argument("--max-node-freq", type=int, default=16)
    parser.add_argument("--max-edge-freq", type=int, default=16)
    parser.add_argument("--max-degree", type=int, default=512)
    parser.add_argument("--freq-embedding-size", type=int, default=16)
    parser.add_argument("--degree-embedding-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layer", type=int, default=5, help="gnn layers")
    parser.add_argument("--set2set-lstm-layer", type=int, default=3, help="lstm layers for s2s")
    parser.add_argument("--set2set-iter", type=int, default=6, help="s2s iteration")
    parser.add_argument("--norm", action="store_true", default=True, help="apply 2-norm on output feats")
    parser.add_argument("--model", type=str, default="gin", choices=["gat", "mpnn", "gin"])
    parser.add_argument("--nce-k", type=int, default=32)
    parser.add_argument("--nce-t", type=float, default=0.07)
    parser.add_argument("--rw-hops", type=int, default=256)
    args = parser.parse_args()

    data_name = 'cora'
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print('device:', device)


    num_nodes = 0
    tit_list = []
    with open('./data/train_text.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            tit_list.append(line[2])
            num_nodes += 1

    print('num_nodes', num_nodes)

    raw_edge_index = [[], []]
    with open('./data/mapped_edges.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            raw_edge_index[0].append(int(line[0]))
            raw_edge_index[1].append(int(line[1]))

    print('num of edges', len(raw_edge_index[0] + raw_edge_index[1]))

    edge_index = [raw_edge_index[0] + raw_edge_index[1], raw_edge_index[1] + raw_edge_index[0]]
    arr_edge_index = np.array(edge_index)
    edge_index = np.array(edge_index)
    edge_index = torch.from_numpy(edge_index).to(device)

    node_f = np.load('./data/llama2_cora_logits.npy')
    node_f = preprocessing.StandardScaler().fit_transform(node_f)
    node_f_numpy = node_f
    node_f = torch.from_numpy(node_f).float().to(device)
    
    
    start = time.perf_counter()

    seed = 1
    main(args)

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
