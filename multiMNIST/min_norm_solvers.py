# This code is from
# Multi-Task Learning as Multi-Objective Optimization
# Ozan Sener, Vladlen Koltun
# Neural Information Processing Systems (NeurIPS) 2018 
# https://github.com/intel-isl/MultiObjectiveOptimization

import numpy as np
import torch


# 最小规范元素求解器
class MinNormSolver:
    MAX_ITER = 250  # 最大迭代次数
    STOP_CRIT = 1e-5  # 停止迭代的阈值

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        用于计算两个向量线性组合的最小规范元素。
        该方法接受三个参数：v1v1、v1v2和v2v2，分别表示两个向量之间的内积。
        根据内积的取值，可以分为三种情况进行计算，得到最小规范元素的系数gamma和对应的代价cost。
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
        cost = v2v2 + gamma*(v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        """
        用于在二维空间中找到最小规范解。
        该方法接受两个参数：vecs和dps。vecs是一个向量列表，dps是一个字典，用于存储向量之间的内积。
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i+1,len(vecs)):
                if (i,j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,j)] += torch.dot(vecs[i][k], vecs[j][k]).item()#torch.dot(vecs[i][k], vecs[j][k]).data[0]
                    dps[(j, i)] = dps[(i, j)]
                if (i,i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,i)] += torch.dot(vecs[i][k], vecs[i][k]).item()#torch.dot(vecs[i][k], vecs[i][k]).data[0]
                if (j,j) not in dps:
                    dps[(j, j)] = 0.0   
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.dot(vecs[j][k], vecs[j][k]).item()#torch.dot(vecs[j][k], vecs[j][k]).data[0]
                c,d = MinNormSolver._min_norm_element_from2(dps[(i,i)], dps[(i,j)], dps[(j,j)])
                if d < dmin:
                    dmin = d
                    sol = [(i,j),c,d]
        return sol, dps

    def _projection2simplex(y):
        """
        实现了一个投影操作，将向量y投影到简单形式（simplex）上。
        简单形式是指满足约束条件的向量，即元素之和等于1，且每个元素的取值范围在[0, 1]之间。
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)  # 获取向量y的长度m
        sorted_y = np.flip(np.sort(y), axis=0)  # 将y按降序排列得到sorted_y
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0)/m
        for i in range(m-1):  # 遍历sorted_y中的元素
            tmpsum+= sorted_y[i]  # 累加到tmpsum中
            tmax = (tmpsum - 1)/ (i+1.0)  # 计算当前的tmax值
            if tmax > sorted_y[i+1]:  # 如果tmax大于sorted_y的下一个元素
                tmax_f = tmax  # 更新tmax_f为tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))
    
    def _next_point(cur_val, grad, n):
        # 计算出一个新的点作为更新点，用于更新最小范数元素的求解。
        # 输入参数包括当前点cur_val、梯度grad和向量长度n。
        proj_grad = grad - ( np.sum(grad) / n )  # 计算投影梯度proj_grad，即将grad减去grad的平均值
        tm1 = -1.0*cur_val[proj_grad<0]/proj_grad[proj_grad<0]
        tm2 = (1.0 - cur_val[proj_grad>0])/(proj_grad[proj_grad>0])
        
        skippers = np.sum(tm1<1e-7) + np.sum(tm2<1e-7)
        t = 1
        if len(tm1[tm1>1e-7]) > 0:
            t = np.min(tm1[tm1>1e-7])
        if len(tm2[tm2>1e-7]) > 0:
            t = min(t, np.min(tm2[tm2>1e-7]))

        next_point = proj_grad*t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs):
        """
        这个函数实现了在凸包中找到最小范数元素的求解。输入参数为向量列表vecs。
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        函数首先找到最佳的两个点的解，并运行【投影梯度下降算法】直到收敛。
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)  # 找到最佳的两个点的解 init_sol，并将计算得到的内积保存到 dps 中。
        
        n=len(vecs)
        sol_vec = np.zeros(n)
        # 将 init_sol 中的解应用到 sol_vec 上，使得 sol_vec 是一个线性组合的形式，满足 u = sum(c_i * vecs[i]) 的约束条件。
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]
    
        iter_count = 0
        # 初始化迭代计数器 iter_count 和梯度矩阵 grad_mat。
        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]  # 梯度矩阵的元素是从字典 dps 中获取的内积值。
                

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0*np.dot(grad_mat, sol_vec)  # 通过将梯度矩阵 grad_mat 与解 sol_vec 做矩阵乘法，并乘以 -1.0，得到梯度方向。
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)  # 根据当前解 sol_vec 和梯度方向 grad_dir 计算新的更新点 new_point。
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            # 重新计算内积用于线性搜索。这里通过两个嵌套的循环遍历 sol_vec 和 new_point 中的元素，并根据内积公式计算 v1v1、v1v2 和 v2v2。
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i]*sol_vec[j]*dps[(i,j)]
                    v1v2 += sol_vec[i]*new_point[j]*dps[(i,j)]
                    v2v2 += new_point[i]*new_point[j]*dps[(i,j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec + (1-nc)*new_point
            # 计算新的解 new_sol_vec，根据最小范数元素的系数 nc 将当前解 sol_vec 与更新点 new_point 进行线性组合。
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        首先找到最佳的两个解，然后运行 【Frank Wolfe】 直到收敛。
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n=len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))
            # 在每次迭代中，首先找到梯度矩阵 `grad_mat` 与当前解 `sol_vec` 的乘积中的最小元素的索引 `t_iter`。

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]  # 梯度矩阵 `grad_mat` 的第 `t_iter` 行、第 `t_iter` 列的元素 `v2v2`

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)  # 计算出最小范数元素的系数 `nc` 和最小范数 `nd`。
            new_sol_vec = nc*sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec


def gradient_normalizers(grads, losses, normalization_type):
    """
    用于计算梯度的归一化因子
    grads 字典，包含了不同参数的梯度；
    losses 字典，包含了不同参数的损失值；
    normalization_type 是一个字符串，表示归一化类型。
    """
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data[0] for gr in grads[t]]))
            # 计算对应梯度 grads[t] 中所有元素的平方和的平方根，并将结果赋值给 gn[t]。
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data[0] for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn