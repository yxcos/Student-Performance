import os
import torch
import random
import argparse
import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log_string(log, string):
    """打印log"""
    log.write(string + '\n')
    log.flush()
    print(string)


def count_parameters(model):
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_seed(seed):
    """Disable cudnn to maximize reproducibility 禁用cudnn以最大限度地提高再现性"""
    torch.cuda.cudnn_enabled = False
    
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


"""图相关"""


def get_adjacency_matrix(distance_df_filename, num_of_vertices, type_='connectivity', id_filename=None):
    
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 建立映射列表
        df = pd.read_csv(distance_df_filename)
        for row in df.values:
            if len(row) != 3:
                continue
            i, j = int(row[0]), int(row[1])
            A[id_dict[i], id_dict[j]] = 1
            A[id_dict[j], id_dict[i]] = 1

        return A

    df = pd.read_csv(distance_df_filename)
    for row in df.values:
        if len(row) != 3:
            continue
        i, j, distance = int(row[0]), int(row[1]), float(row[2])
        if type_ == 'connectivity':
            A[i, j] = 1
            A[j, i] = 1
        elif type == 'distance':
            A[i, j] = 1 / distance
            A[j, i] = 1 / distance
        else:
            raise ValueError("type_ error, must be "
                             "connectivity or distance!")

    return A


def construct_st_adj(multi_order_adj,num_time_steps):
    multi_order_st_adj=[]
    N = multi_order_adj[0].shape[0] 

 
    for k in range(len(multi_order_adj)):
        st_adj = np.zeros((N * num_time_steps, N * num_time_steps)) 
        for i in range(num_time_steps):
            for j in range(i+1):
                st_adj[j * N: (j + 1) * N, i * N: (i + 1) * N] = multi_order_adj[k]   
        multi_order_st_adj.append(st_adj)    
    
    return torch.FloatTensor(multi_order_st_adj)


def construct_multi_order_adj(original_A,pearson_adj,num_spatial_order):

    num_nodes = original_A.shape[0] 
    inclusive_adj=pearson_adj
    for i in range(num_nodes):
        for j in range(i+1):
            if inclusive_adj[i,j] ==0 and original_A[i,j]!=0:
                inclusive_adj[i,j]=original_A[i,j]
                inclusive_adj[j,i]=original_A[i,j]


    multi_order_adj=[] 
    multi_order_adj.append(inclusive_adj) 
    for k in range(1,num_spatial_order):
        k_order_adj=np.zeros((num_nodes,num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if multi_order_adj[k-1][i,j] !=0:
                    for l in range(num_nodes):
                        if multi_order_adj[k-1][j,l] !=0:
                            k_order_adj[i,l]= (multi_order_adj[k-1][i,j]+multi_order_adj[k-1][j,l])/2 
                            k_order_adj[l,i]= (multi_order_adj[k-1][i,j]+multi_order_adj[k-1][j,l])/2
        multi_order_adj.append(k_order_adj)

    return multi_order_adj

def construct_pearson(dataset):
     
    save_path='./garage/{}/pearson_adj.npy'.format(dataset) 

    if os.path.exists(save_path): 
        pearson_adj=np.load(save_path)
        print('pearson_adj loaded')
    else:
        x_path='./data/{}/{}.npz'.format(dataset,dataset) 
        data = normalize(np.load(x_path)['data'])[:, :, 0] 
        num_nodes=data.shape[1]

        pearson=np.corrcoef(data.T)
        pearson_adj = np.zeros([num_nodes,num_nodes])
        adj_percent = 0.01
        top = int(num_nodes * adj_percent)
    
        for i in range(pearson.shape[0]):
            a = pearson[i,:].argsort()[::-1][0:top] 
            for j in range(top):
                pearson_adj[i, a[j]] = pearson[i, a[j]] 
                pearson_adj[a[j], i] = pearson[i, a[j]]
                  
            for k in range(num_nodes):
                if( i==k):
                    pearson_adj[i][k] = 1
        print("The calculation of {} pearson_adj is done!".format(dataset))  
        print(pearson_adj.shape) 
        np.save(save_path,pearson_adj)
    return pearson_adj

def normalize(a):
    mu=np.mean(a,axis=1,keepdims=True)
    std=np.std(a,axis=1,keepdims=True)
    return (a-mu)/std


def construct_adj(A, steps):
    
    N = len(A)  # 获得行数
    adj = np.zeros((N * steps, N * steps))

    for i in range(steps):
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1

    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


"""数据加载器"""


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
       
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """标准转换器"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class NScaler:
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class MinMax01Scaler:
    """最大最小值01转换器"""
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class MinMax11Scaler:
    """最大最小值11转换器"""
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min


def load_dataset(dataset_dir, normalizer, batch_size, valid_batch_size=None, test_batch_size=None, column_wise=False):
    
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    if normalizer == 'max01':
        if column_wise:
            minimum = data['x_train'].min(axis=0, keepdims=True)
            maximum = data['x_train'].max(axis=0, keepdims=True)
        else:
            minimum = data['x_train'].min()
            maximum = data['x_train'].max()

        scaler = MinMax01Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax01 Normalization')

    elif normalizer == 'max11':
        if column_wise:
            minimum = data['x_train'].min(axis=0, keepdims=True)
            maximum = data['x_train'].max(axis=0, keepdims=True)
        else:
            minimum = data['x_train'].min()
            maximum = data['x_train'].max()

        scaler = MinMax11Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax11 Normalization')

    elif normalizer == 'std':
        if column_wise:
            mean = data['x_train'].mean(axis=0, keepdims=True)  
            std = data['x_train'].std(axis=0, keepdims=True)
        else:
            mean = data['x_train'].mean()
            std = data['x_train'].std()

        scaler = StandardScaler(mean, std)
        print('Normalize the dataset by Standard Normalization')

    elif normalizer == 'None':
        scaler = NScaler()
        print('Does not normalize the dataset')
    else:
        raise ValueError

    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler

    return data


"""指标"""


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)

    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)


    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()

    return mae, mape, rmse


if __name__ == '__main__':
    print('performance prediction test')

