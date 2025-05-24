import torch
import numpy as np
import scipy.sparse as sp

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def construct_hypergraph(cascades, user_size, window_size=5):
    """
    Construct hypergraph from cascade data
    
    Args:
        cascades: List of cascades, each cascade is a list of user IDs
        user_size: Total number of users
        window_size: Size of sliding window for dependency hypergraph
        
    Returns:
        HG_Item: Item-based hypergraph (interest)
        HG_User: User-based hypergraph (dependency)
    """
    # User context dictionary for dependency hypergraph
    user_context = {}
    for i in range(user_size):
        user_context[i] = []
    
    # Build context using sliding window
    for cascade in cascades:
        if len(cascade) < window_size:
            # For short cascades, use the entire cascade as context
            for idx in cascade:
                user_context[idx] = list(set(user_context[idx] + cascade))
            continue
            
        # Use sliding window for longer cascades
        for j in range(len(cascade) - window_size + 1):
            window = cascade[j:j + window_size]
            for idx in window:
                user_context[idx] = list(set(user_context[idx] + window))
    
    # Construct User-based hypergraph (dependency)
    indptr, indices, data = [], [], []
    indptr.append(0)
    idx = 0
    
    for user_id in user_context:
        if len(user_context[user_id]) == 0:
            idx += 1
            continue
            
        source = np.unique(user_context[user_id])
        length = len(source)
        s = indptr[-1]
        indptr.append(s + length)
        
        for i in range(length):
            indices.append(source[i])
            data.append(1)
    
    H_User = sp.csr_matrix((data, indices, indptr), shape=(len(user_context) - idx, user_size))
    
    # Normalize User hypergraph
    H_User_sum = 1.0 / H_User.sum(axis=1).reshape(1, -1)
    H_User_sum[H_User_sum == float("inf")] = 0
    
    BH_T = H_User.T.multiply(H_User_sum)
    BH_T = BH_T.T
    H = H_User.T
    
    H_sum = 1.0 / H.sum(axis=1).reshape(1, -1)
    H_sum[H_sum == float("inf")] = 0
    
    DH = H.T.multiply(H_sum)
    DH = DH.T
    HG_User = np.dot(DH, BH_T).tocoo()
    
    # Construct Item-based hypergraph (interest)
    indptr, indices, data = [], [], []
    indptr.append(0)
    
    for cascade in cascades:
        items = np.unique(cascade)
        length = len(items)
        
        s = indptr[-1]
        indptr.append(s + length)
        
        for i in range(length):
            indices.append(items[i])
            data.append(1)
    
    H_Item = sp.csr_matrix((data, indices, indptr), shape=(len(cascades), user_size))
    
    # Normalize Item hypergraph
    H_Item_sum = 1.0 / H_Item.sum(axis=1).reshape(1, -1)
    H_Item_sum[H_Item_sum == float("inf")] = 0
    
    BH_T = H_Item.T.multiply(H_Item_sum)
    BH_T = BH_T.T
    H = H_Item.T
    
    H_sum = 1.0 / H.sum(axis=1).reshape(1, -1)
    H_sum[H_sum == float("inf")] = 0
    
    DH = H.T.multiply(H_sum)
    DH = DH.T
    HG_Item = np.dot(DH, BH_T).tocoo()
    
    # Convert to PyTorch sparse tensors
    HG_Item = sparse_mx_to_torch_sparse_tensor(HG_Item)
    HG_User = sparse_mx_to_torch_sparse_tensor(HG_User)
    
    return HG_Item, HG_User 