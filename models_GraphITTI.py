import torch
import torch.nn as nn
import torch.nn.parameter
from torch_sparse import SparseTensor
from torch_geometric.nn import GraphConv, GCNConv, TransformerConv
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Homo_Att(torch.nn.Module):
    # Modified from https://github.com/chuxuzhang/KDD2019_HetGNN after removing the modules with diff types of attributes
    # renamed to homo as its not really hetro for our data. 
    def __init__(self, in_feats, hidden_size, k):
        super(Homo_Att, self).__init__()
        self.num_attr = 3
        self.num_nodes = 4
        self.embed_d = 32
        self.fc = nn.Linear(32, 1)
        self.fc_mdp = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.a_content_rnn = nn.LSTM(self.embed_d, int(self.embed_d/2), 1, bidirectional = True)
        self.v_neigh_att = nn.Parameter(torch.ones(self.embed_d , 1), requires_grad = True)
        self.softmax = nn.Softmax(dim = 1)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(p = 0.5)
        # self.bn = nn.BatchNorm1d(embed_d)

        # To uniform attribute dimensions
        self.att_dim1 = nn.Linear(6, self.embed_d)
        self.att_dim2 = nn.Linear(6, self.embed_d)
        self.att_dim3 = nn.Linear(17, self.embed_d)
        self.att_dim4 = nn.Linear(12, self.embed_d)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                #nn.init.normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # # Handling non uniform attributes
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        x1 = self.att_dim1(x1)
        x2 = self.att_dim2(x2)
        x3 = self.att_dim3(x3)

        x = torch.reshape(torch.stack([x1,x2,x3]),(x1.shape[0], self.num_attr, x1.shape[1]))

        all_state, last_state = self.a_content_rnn(x)

        q = self.v_neigh_att.unsqueeze(0).expand(len(x),\
             *self.v_neigh_att.size())

        atten_w = self.act(torch.bmm(all_state, self.v_neigh_att.unsqueeze(0).expand(len(x),\
             *self.v_neigh_att.size())))

        atten_w = self.softmax(atten_w).view(len(x), 1, self.num_attr)

        weight_agg_batch = torch.bmm(atten_w, all_state).view(len(x), self.embed_d)

        x1 = self.fc(weight_agg_batch)
        x_mdp = self.fc_mdp(weight_agg_batch)

        return x1,x_mdp

class IntraInterAtt(torch.nn.Module):
    def __init__(self, num_attr,num_nodes, embed_d, device):
        super(IntraInterAtt, self).__init__()
        self.device = device
        self.num_attr = num_attr 
        self.num_nodes = num_nodes 
        self.embed_d = embed_d 
        self.fc = nn.Linear(self.embed_d, 1)
        self.fc_mdp = nn.Linear(self.embed_d, 1)

        self.dropout = nn.Dropout(0.3)
        self.a_content_rnn = nn.LSTM(self.embed_d, int(self.embed_d/2), 1, bidirectional = True,batch_first=True)

        # To uniform attribute dimensions
        self.att_dim1 = nn.Linear(17, self.embed_d)# AuDiff
        self.att_dim2 = nn.Linear(17, self.embed_d) # AUMean
        self.att_dim3 = nn.Linear(6, self.embed_d) # headpose
        self.att_dim4 = nn.Linear(6, self.embed_d) # Gaze
        self.att_dim5_1 = nn.Linear(96961, 1024) #122962 MPII, 96961 ELEA
        self.att_dim5_2 = nn.Linear(1024, self.embed_d)
        # self.att_dim6 = nn.Linear(876, self.embed_d)

        self.intraTrans = TransformerConv(embed_d, embed_d)
        self.intraTrans1 = TransformerConv(embed_d, embed_d)
        self.intraGConv = GCNConv(embed_d, embed_d)
        self.intraGConv1 = GCNConv(embed_d, embed_d)

        self.interTrans = TransformerConv(embed_d, embed_d)
        self.interTrans1 = TransformerConv(embed_d, embed_d)
        self.interGConv = GCNConv(embed_d, embed_d)
        self.interGConv1 = GCNConv(embed_d, embed_d)


    def forward(self, data):

        x, edge_index, [edge_wt_gaze, edge_wt_sp], batch = data.x, data.edge_index, data.edge_weight, data.batch

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]

        x1 = self.att_dim1(x1)
        x2 = self.att_dim2(x2)
        x3 = self.att_dim3(x3)
        x4 = self.att_dim4(x4)
        x5 = self.att_dim5_2(self.att_dim5_1(x5))

        x = torch.reshape(torch.stack([x1,x2,x3,x4, x5]),(x1.shape[0], self.num_attr, x1.shape[1]))

        x, last_state = self.a_content_rnn(x)
        intra_feat = torch.zeros((x.shape[0],x.shape[1], x.shape[2]), dtype=torch.float64)
        for i in range(len(x)):
            x_a = x[i,:,:]

            # Fully connected edges between attributes
            edge_a = torch.tensor([[0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4], #,5,5,5,5,5,5  ,0,1,2,3,4,5
                [0,1,2,3,4, 0,1,2,3,4, 0,1,2,3,4, 0,1,2,3,4, 0,1,2,3,4]], dtype=torch.long).to(self.device)
            
            if edge_a.shape[1] != self.num_attr*self.num_attr:
                print('Incorrect edge weights? Found edges -', edge_a.shape[1])
                return
            x_a = self.intraGConv(x_a,edge_a)
            x_a = self.intraTrans1(x_a,edge_a)

            intra_feat[i,:,:] = x_a

        intra_feat = torch.mean(intra_feat, 1).to(self.device)
        intra_feat = intra_feat.type(torch.float32)

        inter_feat = self.interGConv(intra_feat,edge_index, edge_wt_gaze)

        inter_feat = self.interTrans(inter_feat, edge_index)

        x1 = self.fc(inter_feat)
        x1_mdp = self.fc_mdp(inter_feat)

        return x1, x1_mdp



class GroupDNNIJCAI(torch.nn.Module):
    def __init__(self, num_attr,num_nodes, embed_d):
        super(GroupDNNIJCAI, self).__init__()
        self.num_attr = num_attr 
        self.num_nodes = num_nodes 
        self.embed_d = embed_d 

        self.fc = nn.Linear(self.embed_d, 1)
        self.fc_mdp = nn.Linear(self.embed_d, 1)

        self.interTrans = TransformerConv(embed_d, embed_d)
        self.interTrans1 = TransformerConv(embed_d, embed_d)
        self.interGConv = GCNConv(embed_d, embed_d)
        self.interGConv1 = GCNConv(embed_d, embed_d)

    def forward(self, data):
        # graph on features
        x, edge_index,edge_wt, batch = data.x, data.edge_index,data.edge_weight, data.batch

        x1 = self.att_dim1(x)
        intra_feat = self.interTrans1(x1,edge_index)
        inter_feat = self.interGConv(intra_feat,edge_index, edge_wt)
        inter_feat = self.interTrans(inter_feat, edge_index)
        inter_feat = self.interTrans1(inter_feat, edge_index)
        x1 = self.fc(inter_feat)
        # x1_mdp = self.fc_mdp(inter_feat)
        return x1#, x1_mdp
