import torch
import torch.nn.functional as F
import torch.nn as nn


class Aggregation(nn.Module):
    def __init__(self, adj, in_dim, out_dim, num_vertices, activation='GLU'):
       
        super(Aggregation, self).__init__()
        self.adj = adj
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_vertices = num_vertices
        self.activation = activation

        assert self.activation in {'GLU', 'relu'}

        if self.activation == 'GLU':
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)

    def forward(self, x, mask=None):
        
        adj = self.adj
        if mask is not None:
            adj = adj.to(mask.device) * mask
        x = torch.einsum('nm, mbc->nbc', adj.to(x.device), x)  

        if self.activation == 'GLU':
            lhs_rhs = self.FC(x)  
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1) 

            out = lhs * torch.sigmoid(rhs)
            del lhs, rhs, lhs_rhs

            return out

        elif self.activation == 'relu':
            return torch.relu(self.FC(x))  


class AggregationOperation(nn.Module):
    def __init__(self, adj, in_dim, out_dims, num_of_vertices, activation='GLU'):
        
        super(AggregationOperation, self).__init__()
        self.adj = adj
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation

        self.aggregations = nn.ModuleList()

        self.aggregations.append(
            Aggregation(
                adj=self.adj,
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],
                num_vertices=self.num_of_vertices,
                activation=self.activation
            )
        )

    def forward(self, x, mask=None):
       
        need_concat = []

        for i in range(len(self.out_dims)):
            x = self.aggregations[i](x, mask)
            need_concat.append(x)

        need_concat = [
            torch.unsqueeze(
                h[0: self.num_of_vertices], dim=0 
            ) for h in need_concat
        ]

        out = torch.max(torch.cat(need_concat, dim=0), dim=0).values 

        del need_concat

        return out


class AggregationLayer(nn.Module):
    def __init__(self,
                 adj,
                 history,
                 num_of_vertices,
                 in_dim,
                 out_dims,
                 strides=3,
                 activation='GLU',
                 temporal_emb=True,
                 spatial_emb=True):
       
        super(AggregationLayer, self).__init__()
        self.adj = adj
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices

        self.activation = activation
        self.spatial_emb = spatial_emb

        self.AggregationOperations = nn.ModuleList()
        for i in range(self.history - self.strides + 1):
            self.AggregationOperations.append(
                AggregationOperation(
                    adj=self.adj,
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )

        
        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim))
            

        self.reset()

    def reset(self):
       
        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x, mask=None):
                
        if self.spatial_emb:
            x = x + self.spatial_embedding

        need_concat = []
        batch_size = x.shape[0]

        for i in range(self.history - self.strides + 1):
            t = x[:, i: i+self.strides, :, :] 

            t = torch.reshape(t, shape=[batch_size, self.strides * self.num_of_vertices, self.in_dim])
           
            t = self.AggregationOperations[i](t.permute(1, 0, 2), mask)  

            t = torch.unsqueeze(t.permute(1, 0, 2), dim=1) 

            need_concat.append(t)

        out = torch.cat(need_concat, dim=1) 

        del need_concat, batch_size

        return out


class output_layer(nn.Module):
    def __init__(self, num_of_vertices, history, in_dim,
                 hidden_dim=128, horizon=12):
      
        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.history = history
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        self.FC1 = nn.Linear(self.in_dim * self.history, self.hidden_dim, bias=True)

        self.FC2 = nn.Linear(self.hidden_dim, self.horizon, bias=True)

    def forward(self, x):
       
        batch_size = x.shape[0]

        x = x.permute(0, 2, 1, 3) 

        out1 = torch.relu(self.FC1(x.reshape(batch_size, self.num_of_vertices, -1)))
       

        out2 = self.FC2(out1)  

        del out1, batch_size

        return out2.permute(0, 2, 1) 


class USTGA(nn.Module):
    def __init__(self, adj, history, num_of_vertices, in_dim, hidden_dims,
                 first_layer_embedding_size, out_layer_dim, activation='GLU', use_mask=True,
                 temporal_emb=True, spatial_emb=True, horizon=12, strides=3):
        super(USTGA, self).__init__()
        self.adj = adj
        self.num_of_vertices = num_of_vertices
        self.hidden_dims = hidden_dims
        self.out_layer_dim = out_layer_dim
        self.activation = activation
        self.use_mask = use_mask

        self.spatial_emb = spatial_emb
        self.horizon = horizon
        self.strides = strides

        self.First_FC = nn.Linear(in_dim, first_layer_embedding_size, bias=True)
        self.spatialAttention=spatialAttention(hidden_dims[0][0])
        
        self.AggregationLayers = nn.ModuleList()
        for i in range(len(self.hidden_dims)):
            self.AggregationLayers.append(
                AggregationLayer(
                adj=adj[i],
                history=history,
                num_of_vertices=self.num_of_vertices,
                in_dim=first_layer_embedding_size,
                out_dims=self.hidden_dims[i],
                strides=self.strides,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb
            )    
            )

        self.predictLayer = nn.ModuleList()
        for t in range(self.horizon):
            self.predictLayer.append(
                output_layer(
                    num_of_vertices=self.num_of_vertices,
                    history=4+len(self.hidden_dims)*(4-strides+1),
                    in_dim=first_layer_embedding_size,
                    hidden_dim=out_layer_dim,
                    horizon=1
                )
            )

        if self.use_mask:
            self.mask=[]
            for i in range(len(self.hidden_dims)):
                mask = torch.zeros_like(self.adj[i])
                mask[self.adj[i] != 0] = self.adj[i][self.adj[i] != 0]
                self.mask.append(nn.Parameter(mask)) 
        else:
            self.mask = None

    def forward(self, x):
       
        x = torch.relu(self.First_FC(x))  
        
        multi_order_output=[]
        multi_order_output.append(x) 
        for i in range(len(self.AggregationLayers)):
            x_current_order=x
            if i==1:
                x_current_order=self.spatialAttention(x_current_order)
            x_current_order= self.AggregationLayers[i](x_current_order, self.mask[i])
            multi_order_output.append(x_current_order)
        x=torch.concat(multi_order_output,1) 

        
        need_concat = []
        for i in range(self.horizon):
            out_step = self.predictLayer[i](x)  
            need_concat.append(out_step)

        out = torch.cat(need_concat, dim=1) 

        del need_concat

        return out

class spatialAttention(nn.Module):
    def __init__(self,D):
        super(spatialAttention, self).__init__()
        self.FC_q=nn.Linear(D,D)
        self.FC_k=nn.Linear(D,D)
        self.FC_v=nn.Linear(D,D)

    def forward(self, X):
         
        query = self.FC_q(X)
        key = self.FC_v(X)
        value=self.FC_v(X)
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (X.shape[3] ** 0.5)
        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        del attention
        return X

