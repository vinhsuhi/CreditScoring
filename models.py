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
#     adj = torch.zeros(N, N).float()




class ScoringNetwork(nn.Module):
    """
    Class for initializing the score for each node
    TODO: DONE!!!
    """
    def __init__(self, att_dim):
        super(ScoringNetwork, self).__init__()
        self.att_dim = att_dim
        self.linear1 = nn.Linear(att_dim, int(att_dim * 0.75))
        self.linear2 = nn.Linear(int(att_dim * 0.75), 1)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        # init_weight(self.modules)
    
    def forward(self, input):
        try:
            if input == 'eye':
                input = torch.eye(self.att_dim).float()
        except:
            print("ERROR")
            exit()
        return self.linear2(self.act(self.linear(input)))



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
        TODO: Design for sparse matrix
        """
        N = scores.size(0)
        sum_attention = torch.zeros(N, N)
        cuda = True
        if cuda:
            sum_attention = sum_attention
        for i in tqdm(range(len(edge_type_emb))):
            edge_list = self.adjs[i]
            edge_type_i = edge_type_emb[i]
            edge_type_i_repeated = edge_type_i.repeat(len(edge_list[0])).view(len(edge_list[0]), -1) # n_edgex x emb_dim
            source_scores = scores[edge_list[0]]
            target_scores = scores[edge_list[1]]
            concated_vector = torch.cat([source_scores, edge_type_i_repeated, target_scores], dim=-1)
            attention_i = concated_vector.matmul(self.attention_weight)
            attention_i = attention_i.view(len(edge_list[0]))
            sum_attention[edge_list[0], edge_list[1]] += attention_i

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
    dense = dense.float()
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
    
    degree = torch.FloatTensor(degree)
    att = "eye"

    model = GENI(
                n_edge_types=len(edge_list_dict),
                edge_emb_dim=10,
                num_agg_layer=3,
                deg = degree,
                num_head_attentions=4,
                att_dim=len(entity2index),
                adjs=edge_list_dict
    )

    model = model
    
    real_scores = torch.FloatTensor(np.random.rand(len(entity2index)))

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.005, weight_decay=0.0005)
    for epoch in tqdm(range(50), desc="Training"):
        optimizer.zero_grad()
        scores = model(att)
        loss = loss_function(scores[:10], real_scores[:10])
        print("vvalidate score: {:.4f}".format(loss_function(scores[10:], real_scores[10:])))
        print(loss.data)
        loss.backward()
        optimizer.step()

