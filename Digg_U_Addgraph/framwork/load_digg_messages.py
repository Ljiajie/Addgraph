# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: load_uci messages.py
# @time: 2020/03/08
import numpy as np
import random


def load_digg_messages(data_path, sample_rate):
    """ function load_uci_messages
    #  [data, n, m] = load_uci_message(sample_rate)
    #  load data set uci_message and preprocess it
    #  parameter: sample_rate to subsample edges
    #  return data: network(each row is one edge); n: number of total nodes; m:
    #  number of edges
    """
    # Load Edges from EdgeList
    oedges = np.loadtxt(data_path, dtype=int, comments='%', usecols=(0, 1))
    # region change to undirected graph
    idx_reverse = np.nonzero(oedges[:, 0] - oedges[:, 1] > 0)
    tmp = oedges[idx_reverse]
    tmp[:, [0, 1]] = tmp[:, [1, 0]]
    oedges[idx_reverse] = tmp
    # endregion

    #egion remove self-loops
    idx_remove_dups = np.nonzero(oedges[:, 0] - oedges[:, 1] < 0)
    oedges = oedges[idx_remove_dups]
    oedges = oedges[:, 0:2]
    #endregion

    #region only keep unique edges
    # oedges, ind_ = np.unique(oedges, axis=0, return_index=True)
    #endregion

    m = len(oedges)
    m_ = int(np.floor(m * sample_rate))
    oedges = oedges[0:m_, :]

    # region only keep unique edges
    # oedges, ind_ = np.unique(oedges, axis=0, return_index=True)
    # endregion
    # delete edge
    # oedges = np.array(list(set([tuple(t) for t in oedges])))

    # re-assign id
    unique_id = np.unique(oedges)  # 返回edges中所有唯一的顶点(一维)
    n = len(unique_id)
    _, digg = ismember(oedges, unique_id)

    data = digg

    # np.random.seed(101)
    # np.random.shuffle(data)

    return data, n, m_  # 原始数据，顶点数(一定sample—rate后边中的顶点)，边数


def ismember(a, b_vec):
    """ MATLAB equivalent ismember function """

    shape_a = a.shape

    a_vec = a.flatten()

    bool_ind = np.isin(a_vec, b_vec)
    common = a_vec[bool_ind]
    common_unique, common_inv = np.unique(common, return_inverse=True)  # common = common_unique[common_inv]
    b_unique, b_ind = np.unique(b_vec, return_index=True)  # b_unique = b_vec[b_ind]
    common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]
    flag = bool_ind.reshape(shape_a)
    content = (common_ind[common_inv]).reshape(shape_a) + 1

    return flag, content  # 去除了顶点顺序间的冗余，使得顶点顺序连续(因为数据之前经过sample，边数据中的顶点可能不连续)


if __name__ == "__main__":
    data_path = '../munmun_digg_reply/out.munmun_digg_reply'
    data, n, m = load_digg_messages(data_path, 0.25*1)
    print(data.shape, n, m)  # 1899个顶点，13838条边  (85155, 2) 30360 86203  0.3-(8555, 2) 6547 8620
    print(data[:10, :])
    print(data[-10:, :])
    # fresult = open('membership.txt','w')#w:只写，文件已存在则清空，不存在则创建
    # for i in range(n):
    #     c = random.randint(1, 8)
    #     fresult.write(str(c) + '\n')
    # fresult.close()
    # a=np.array([[1,6],[3,4],[8,5]])
    # b=np.unique(a)
    # m,n=ismember(a, b)
    # print(m,'\n',n)
    Adj = np.zeros((n, n))
    for edge in data:
        Adj[edge[0].item() - 1][edge[1].item() - 1] = Adj[edge[0].item() - 1][edge[1].item() - 1] + 1
        Adj[edge[1].item() - 1][edge[0].item() - 1] = Adj[edge[1].item() - 1][edge[0].item() - 1] + 1
    D = Adj.sum(1)
    print(D)
    print(D.max())