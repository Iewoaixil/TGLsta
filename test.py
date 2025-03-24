import numpy as np
import argparse
import torch
from random import sample
import random
import math
import time
from model import CLIP, tokenize
from torch import nn, optim
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from multitask import multitask_data_generator
from model_g_coop import CoOp
import json
from data_graph import DataHelper
from torch.utils.data import DataLoader
import os
import time
import dgl
import numpy as np
import torch
from utils import batcher
from graph_encoder import GraphEncoder
import math
from utils import _rwr_trace_to_dgl_graph
import warnings
from collections import defaultdict, namedtuple
warnings.filterwarnings("ignore")

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

def generate_node_s_f(args_test):
    print("=> loading checkpoint '{}'".format(args_test.load_path))
    checkpoint = torch.load(args_test.load_path, map_location="cpu")
    args = checkpoint["opt"]
    assert args_test.gpu is None or torch.cuda.is_available()
    print("Use GPU: {} for generation".format(args_test.gpu))
    args.gpu = args_test.gpu
    args.device = torch.device("cpu") if args.gpu is None else torch.device(args.gpu)
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
    graph = dgl.DGLGraphStale()
    src, dst = arr_edge_index
    graph.add_nodes(max(src.max(), dst.max()) + 1)
    graph.add_edges(src, dst)
    model = GraphEncoder(
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
    model = model.to(args.device)
    model.load_state_dict(checkpoint["model"])
    print("Done loading pre-trained model from cached files.")

    del checkpoint
    print("start generating the embeddings of each node")
    model.eval()
    emb_list = []
    for nodes in graph.nodes():
        graph_q, graph_k = generate_embeddings(graph, nodes, rw_hops=args.rw_hops)
        graph_q = graph_q.to(args.device)
        graph_k = graph_k.to(args.device)
        with torch.no_grad():
            feat_q = model(graph_q) 
            feat_k = model(graph_k) 
        emb_list.append(((feat_q + feat_k) / 2).detach().cpu())
    emb = torch.cat(emb_list)
    print("Done, saving file: ", args_test.saved_name)
    np.save(args_test.saved_name, emb.numpy())   
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    setup_seed(seed)
    clip_model = CLIP(args)
    clip_model.load_state_dict(torch.load('./res/{}/model.pkl'.format(data_name), map_location=device))
    task_list, train_idx, val_idx, test_idx = multitask_data_generator(lab_list, labeled_ids, labels, args.k_spt,
                                                                       args.k_val, args.k_qry, args.n_way)
    all_acc = []
    f1_list = []
    generate_node_s_f(parser.parse_args())
    node_s_f = np.load(args.saved_name)
    node_s_f = torch.from_numpy(node_s_f).to(device)
    for j in range(len(task_list)):

        train_idx_ts = torch.from_numpy(np.array(train_idx[j])).to(device)
        val_idx_ts = torch.from_numpy(np.array(val_idx[j])).to(device)
        test_idx_ts = torch.from_numpy(np.array(test_idx[j])).to(device)

        train_truth = np.array(lab_list)[np.array(train_idx[j])]
        val_truth = np.array(lab_list)[np.array(val_idx[j])]
        test_truth = np.array(lab_list)[np.array(test_idx[j])]

        task_lables_arr = np.array(labels)[task_list[j]]
        task_labels_dict = dict()
        for i in range(task_lables_arr.shape[0]):
            task_labels_dict[task_lables_arr[i]] = i

        train_truth_ts = [task_labels_dict[train_truth[i]] for i in range(len(train_truth))]
        train_truth_ts = torch.from_numpy(np.array(train_truth_ts)).to(device)

        val_truth_ts = [task_labels_dict[val_truth[i]] for i in range(len(val_truth))]
        val_truth_ts = torch.from_numpy(np.array(val_truth_ts)).to(device)

        test_truth_ts = [task_labels_dict[test_truth[i]] for i in range(len(test_truth))]
        test_truth_ts = torch.from_numpy(np.array(test_truth_ts)).to(device)

        task_lables = task_lables_arr.tolist()
        Data = DataHelper(arr_edge_index, args, train_idx[j])
        loader = DataLoader(Data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        for i_batch, sample_batched in enumerate(loader):
            s_n = sample_batched['s_n'].numpy()
            t_n = sample_batched['t_n'].numpy()
        s_n = s_n.reshape(args.num_labels, args.k_spt)
        t_n = t_n.reshape(args.num_labels, args.k_spt * args.neigh_num)
        temp = []
        for i in range(args.num_labels):
            temp.append(np.concatenate((s_n[i], t_n[i])))
        g_texts = []
        for i in range(len(temp)):
            g_text = [tit_list[a] for a in temp[i]]
            g_texts.append(g_text)

        model = CoOp(args, task_lables, clip_model, g_texts, device)

        best_val = 0
        patience = 10
        counter = 0
    
        for epoch in range(1, args.ft_epoch + 1):
            model.train()
            train_logits = model.forward(train_idx_ts, node_f, node_s_f, edge_index, train_truth_ts)

            model.eval()
            with torch.no_grad():
                res = model.forward(val_idx_ts, node_f, node_s_f, edge_index, val_truth_ts, training=False)
                
                val_acc = accuracy_score(val_truth_ts.cpu(), res.argmax(dim=1).cpu())
                if val_acc <= best_val:
                    counter += 1
                    if counter >= patience:
                        break
                else:
                    best_val = val_acc
                    torch.save(model, './res/{}/g_coop.pkl'.format(data_name))
                    counter = 0
        best_model = torch.load('./res/{}/g_coop.pkl'.format(data_name))
        best_model.eval()
        with torch.no_grad():
            res = model.forward(test_idx_ts, node_f, node_s_f, edge_index, test_truth_ts, training=False)
            test_acc = accuracy_score(test_truth_ts.cpu(), res.argmax(dim=1).cpu())
            all_acc.append(test_acc)
            f1 = f1_score(test_truth_ts.cpu(), res.argmax(dim=1).cpu(), average='macro')
            f1_list.append(f1)

    ans = round(np.mean(all_acc).item(), 4)
    print('acc', ans)

    ans = round(np.mean(f1_list).item(), 4)
    print('macro f1', ans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--ft_epoch', type=int, default=50, help='fine-tune epoch')
    parser.add_argument('--lr', type=float, default=2e-5)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gnn_input', type=int, default=4096)
    parser.add_argument('--gnn_hid', type=int, default=128)
    parser.add_argument('--gnn_output', type=int, default=64)

    parser.add_argument('--edge_coef', type=float, default=0.1)
    parser.add_argument('--neigh_num', type=int, default=3)

    parser.add_argument('--num_labels', type=int, default=5)
    parser.add_argument('--k_spt', type=int, default=5)
    parser.add_argument('--k_val', type=int, default=5)
    parser.add_argument('--k_qry', type=int, default=50)
    parser.add_argument('--n_way', type=int, default=5)

    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--coop_n_ctx', type=int, default=4)
    parser.add_argument('--prompt_lr', type=float, default=0.01)

    parser.add_argument('--position', type=str, default='end')
    parser.add_argument('--class_specific', type=bool, default=True)
    parser.add_argument('--ctx_init', type=bool, default=False)

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--load-path", type=str, default="./res/cora/model_gcc.pth")
    parser.add_argument("--saved-name", default="./res/cora/cora.npy")
    args = parser.parse_args()

    data_name = 'cora'
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print('device:', device)
    FType = torch.FloatTensor
    LType = torch.LongTensor

    num_nodes = 0
    tit_list = []
    lab_list = []
    with open('./data/train_text.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            tit_list.append(line[2])
            lab_list.append(line[3])
            num_nodes += 1

    print('num_nodes', num_nodes)

    labeled_ids = []
    for i in range(len(lab_list)):
        if lab_list[i] != 'nan':
            labeled_ids.append(i)

    print('{} nodes having lables'.format(len(labeled_ids)))

    raw_edge_index = [[], []]
    with open('./data/mapped_edges.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            raw_edge_index[0].append(int(line[0]))
            raw_edge_index[1].append(int(line[1]))

    edge_index = [raw_edge_index[0] + raw_edge_index[1], raw_edge_index[1] + raw_edge_index[0]]
    arr_edge_index = np.array(edge_index)
    edge_index = np.array(edge_index)
    edge_index = torch.from_numpy(edge_index).to(device)

    node_f = np.load('./data/llama2_cora_logits.npy')
    node_f = preprocessing.StandardScaler().fit_transform(node_f)
    node_f = torch.from_numpy(node_f).float().to(device)

    with open('./data/lab_list.txt', 'r') as f:
        line = f.readline().strip().split('\t')
        label_texts = line

    labels = []
    for i in label_texts:
        if i != 'nan':
            labels.append(i)

    start = time.perf_counter()
    all_acc_list = []
    all_macf1_list = []

    seed = 1
    print('seed', seed)
    main(args)
    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
