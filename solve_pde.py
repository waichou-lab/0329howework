import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

def solve_pde(m):
    """
    求解橢圓型 PDE，返回誤差 2-範數和數值解矩陣
    """
    h = 1.0 / m
    x_coord = np.linspace(0, 1, m+1)
    y_coord = np.linspace(0, 1, m+1)
    X, Y = np.meshgrid(x_coord, y_coord, indexing='ij')
    
    # 係數矩陣
    a = Y * np.exp((X**2 + Y**2)/2)
    b = X * np.exp((X**2 + Y**2)/2)
    f = X * Y * (X + Y) * (X * Y - 3)
    
    # 內部點索引映射
    N = (m-1)**2
    idx_map = np.zeros((m+1, m+1), dtype=int)
    idx = 0
    for i in range(1, m):
        for j in range(1, m):
            idx_map[i, j] = idx
            idx += 1
    
    # 組裝稀疏矩陣和右端項
    A = lil_matrix((N, N))
    b_vec = np.zeros(N)
    
    for i in range(1, m):
        for j in range(1, m):
            row = idx_map[i, j]
            A[row, row] = -2 * (a[i, j] + b[i, j])
            
            # 左邊界 (i-1, j)
            if i-1 == 0:
                b_vec[row] -= a[i, j] * 0.0
            else:
                A[row, idx_map[i-1, j]] += a[i, j]
            
            # 右邊界 (i+1, j)
            if i+1 == m:
                u_right = y_coord[j] * np.exp(-(1 + y_coord[j]**2)/2)
                b_vec[row] -= a[i, j] * u_right
            else:
                A[row, idx_map[i+1, j]] += a[i, j]
            
            # 下邊界 (i, j-1)
            if j-1 == 0:
                b_vec[row] -= b[i, j] * 0.0
            else:
                A[row, idx_map[i, j-1]] += b[i, j]
            
            # 上邊界 (i, j+1)
            if j+1 == m:
                u_top = x_coord[i] * np.exp(-(x_coord[i]**2 + 1)/2)
                b_vec[row] -= b[i, j] * u_top
            else:
                A[row, idx_map[i, j+1]] += b[i, j]
            
            b_vec[row] += h**2 * f[i, j]
    
    # 求解線性系統
    A_csr = csr_matrix(A)
    u_internal = spsolve(A_csr, b_vec)
    
    # 重構完整解矩陣
    u_num = np.zeros((m+1, m+1))
    for i in range(1, m):
        for j in range(1, m):
            u_num[i, j] = u_internal[idx_map[i, j]]
    
    # 邊界條件賦值
    u_num[0, :] = 0.0
    u_num[:, 0] = 0.0
    for i in range(m+1):
        u_num[i, m] = x_coord[i] * np.exp(-(x_coord[i]**2 + 1)/2)
    for j in range(m+1):
        u_num[m, j] = y_coord[j] * np.exp(-(1 + y_coord[j]**2)/2)
    
    # 真解
    u_true = X * Y * np.exp(-(X**2 + Y**2)/2)
    
    # 誤差 2-範數（排除左邊界和下邊界）
    error = u_num - u_true
    E = error[1:, 1:]
    norm2 = np.sqrt(h * h * np.sum(E**2))
    
    return norm2, u_num