import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm


def init_weight(modules, activation=None):
    """
    Weight initialization
    :param modules: Iterable of modules
    :param activation: Activation function.
    """
    for m in modules:
        if isinstance(m, nn.Linear):
            if activation is None:
                m.weight.data = init.xavier_uniform_(m.weight.data) #, gain=nn.init.calculate_gain(activation.lower()))
            else:
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain(activation.lower()))
            if m.bias is not None:
                m.bias.data = init.constant_(m.bias.data, 0.0)
            

# def get_ith_adj(predicates, edge_path, entity_to_index, relation_index):
#     N = len(entity_to_index)
#     adj = torch.zeros(N, N).float().cuda()




class ScoringNetwork(nn.Module):
    """
    Class for initializing the score for each node
    TODO: DONE!!!
    """
    def __init__(self, att_dim):
        super(ScoringNetwork, self).__init__()
        self.att_dim = att_dim
        self.linear = nn.Linear(att_dim, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        # init_weight(self.modules)
    
    def forward(self, input):
        try:
            if input == 'eye':
                input = torch.eye(self.att_dim).float().cuda()
        except:
            print("ERROR")
            exit()
        return self.linear(input)



class ScoreAggregation(nn.Module):
    """
    Given Averaged Score at last layer, compute score at this layer
    TODO: DONE!
    """
    def __init__(self, num_head_attentions, adjs, edge_emb_dim, edge_type_emb):
        super(ScoreAggregation, self).__init__()
        self.num_head_attentions = num_head_attentions
        self.attentions = []
        self.edge_type_emb = edge_type_emb
        for i in range(num_head_attentions):
            self.attentions.append(SelfAttention(adjs, edge_emb_dim))
        self.attentions = nn.ModuleList(self.attentions)


    def forward(self, scores):
        sum_scores = 0
        for i in range(self.num_head_attentions):
            scores_i = self.attentions[i](scores, self.edge_type_emb)
            sum_scores += scores_i
        new_scores = sum_scores / self.num_head_attentions
        return new_scores



class SelfAttention(nn.Module):
    """
    given scores of last layer
    TODO: DONE!
    """
    def __init__(self, adjs, edge_emb_dim):
        super(SelfAttention, self).__init__()
        self.adjs = adjs
        self.attention_weight = nn.Parameter(torch.Tensor(edge_emb_dim + 2, 1))
        nn.init.xavier_normal_(self.attention_weight.data)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.softmax_layer = nn.Softmax(dim=-1)


    def gen_attention_matrix(self, scores, edge_type_emb):
        """
        TODO: DONE!
        """
        N = scores.size(0)
        repeated_scores = scores.repeat(1, N)
        sum_attention = torch.zeros(N, N, 1)
        zero_matrix = torch.zeros(N, N, edge_type_emb[0].size(-1))
        # if self.adjs[0].is_cuda:
        cuda = True
        if cuda:
            sum_attention = sum_attention.cuda()
            zero_matrix = zero_matrix.cuda()
        for i in range(len(edge_type_emb)):
            adj_i = edge_list_to_dense(adjs[i])
            edge_type_i = edge_type_emb[i]
            adj_i_source = (repeated_scores * adj_i).resize(N, N, 1) # 34 x 34 x 1
            adj_i_target = (repeated_scores.t() * adj_i).resize(N, N , 1)
            adj_i_multi = torch.cat([adj_i.view(N, N, 1)] * len(edge_type_i), dim=-1)
            edge_matrix = edge_type_i.repeat(N, N).view(N, N, -1)
            adj_edge = torch.where(adj_i_multi > 0, edge_matrix, zero_matrix)
            concated_attention = torch.cat([adj_i_source, adj_edge, adj_i_target], dim=-1)
            attention_i = concated_attention.matmul(self.attention_weight)
            sum_attention += attention_i
        # print(self.attention_weight.grad)
        sum_attention = sum_attention.squeeze()
        att = self.act(sum_attention)
        att = self.softmax_layer(att)
        return att
        

    def forward(self, scores, edge_type_emb):
        attention_matrix = self.gen_attention_matrix(scores, edge_type_emb)
        return torch.matmul(attention_matrix, scores)


class CentralityAdjustment(nn.Module):
    def __init__(self, deg):
        super(CentralityAdjustment, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1).fill_(1))
        self.beta = nn.Parameter(torch.Tensor(1).fill_(0))
        self.act = nn.ReLU()
        self.deg = deg
    
    def forward(self, scores):
        epsilon = 1e-10
        centrality = torch.log(self.deg + epsilon)
        flex_centrality = self.gamma * centrality + self.beta
        final_score = self.act(flex_centrality * scores)
        return scores


class GENI(nn.Module):
    def __init__(self, n_edge_types=1, edge_emb_dim=20, num_agg_layer=2, deg=None, num_head_attentions=2, att_dim=0, adjs=[]):
        super(GENI, self).__init__()

        # self.edge_embeddings = nn.Embedding(n_edge_types, edge_emb_dim)
        self.edge_embeddings = nn.Parameter(torch.FloatTensor(n_edge_types, edge_emb_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.edge_embeddings.data)
        self.num_agg_layer = num_agg_layer
        self.initializer = ScoringNetwork(att_dim)
        self.aggregators = []
        for i in range(num_agg_layer):
            self.aggregators.append(ScoreAggregation(num_head_attentions, adjs, edge_emb_dim, self.edge_embeddings))
        self.aggregators = nn.ModuleList(self.aggregators)
        self.centralizer = CentralityAdjustment(deg)

        
    def forward(self, inputs):
        """
        git is the GENI pipeline 
        1. init score
        2. aggregate
        3. centrality adjustment
        """
        # 1 init score:
        scores = self.initializer(inputs)
        # 2 aggregate:
        output_score = None
        for i in range(self.num_agg_layer):
            if i == 0:
                output_score = self.aggregators[i](scores)
            else:
                output_score = self.aggregators[i](output_score)
        
        # 3 centrality adjustment:
        final_score = self.centralizer(output_score)
        return output_score


def loss_function(predicted_scores, real_scores):
    return ((predicted_scores.squeeze() - real_scores) ** 2).mean()


def edge_list_to_dense(edge_list, num_nodes):
    dense = torch.zeros(num_nodes, num_nodes)
    dense[edge_list[0], edge_list[1]] = 1
    dense[edge_list[1], edge_list[0]] = 1
    dense = dense.float().cuda()
    return dense


if __name__ == "__main__":
    """
    what will I do???

    1 - read entity2index == > dict(entity: index) DONE!
    2 - read relation2index == > dict(relation: index) .. DONE!
    3 - read edge file: == > dict(index_of_entity: [[source], [target]]) DONE!
    4 - edge_list to dense: DONE!
    """
    prefix = "dataset/FB15k"

    entity2index_path = "{}/entities_to_index.txt".format(prefix)
    predicates2index_path = "{}/predicates_to_index.txt".format(prefix)
    edge_path = "{}/all_data.txt".format(prefix)

    # 1. Read entity2index
    print("Read entity2index...")
    entity2index = dict()
    with open(entity2index_path, "r", encoding="utf-8") as file:
        for line in file:
            data_line = line.strip().split()
            entity = data_line[0]
            index = data_line[1]
            entity2index[entity] = int(index)
    file.close()

    # 2. Read relation2index
    print("Read predicate2index")
    predicates2index = dict()
    with open(predicates2index_path, "r", encoding="utf-8") as file:
        for line in file:
            data_line = line.strip().split()
            predicate = data_line[0]
            index = data_line[1]
            predicates2index[predicate] = int(index)
    file.close()

    # 3. Read edge file
    print("Read edge file")
    edge_list_dict = dict()
    # degree = {entity2index[entity]: 0 for entity in entity2index.keys()}
    degree = np.zeros(len(entity2index))
    with open(edge_path, "r", encoding="utf-8") as file:
        for line in file:
            e1, r, e2 = line.strip().split()
            e1_index = entity2index[e1]
            e2_index = entity2index[e2]
            degree[e1_index] += 1
            degree[e2_index] += 1
            r_index= predicates2index[r]
            if r_index not in edge_list_dict:
                edge_list_dict[r_index] = [[e1_index], [e2_index]]
            else:
                edge_list_dict[r_index][0].append(e1_index)
                edge_list_dict[r_index][1].append(e2_index)
    
    degree = torch.FloatTensor(degree).cuda()
    att = "eye"

    model = GENI(
                n_edge_types=len(edge_list_dict),
                edge_emb_dim=20,
                num_agg_layer=2,
                deg = degree,
                num_head_attentions=2,
                att_dim=len(entity2index),
                adjs=edge_list_dict
    )
    real_scores = torch.FloatTensor(np.random.rand(len(entity2index))).cuda()

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.01)
    for epoch in tqdm(range(1), desc="Training"):
        optimizer.zero_grad()
        scores = model(att)
        loss = loss_function(scores[:10], real_scores[:10])
        print("vvalidate score: {:.4f}".format(loss_function(scores[10:], real_scores[10:])))
        print(loss.data)
        loss.backward()
        optimizer.step()


    """
    # DEGREE
    degree = degree.cuda()
    # ATT
    att = torch.eye(len(G.nodes())).cuda()
    model = GENI(2, 20, 2, degree, 2, att.size(1), adjs)
    model = model.cuda()
    # real_scores = torch.FloatTensor(np.random.rand(len(G.nodes()))).cuda()
    real_scores = degree / degree.max()

    # optimizer = torch.optim.Adam()
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.01)
    # import pdb
    # pdb.set_trace()
    for epoch in tqdm(range(20), desc="Training"):
        optimizer.zero_grad()
        scores = model(att)
        loss = loss_function(scores[:10], real_scores[:10])
        print("vvalidate score: {:.4f}".format(loss_function(scores[10:], real_scores[10:])))
        print(loss.data)
        loss.backward()
        optimizer.step()

    """