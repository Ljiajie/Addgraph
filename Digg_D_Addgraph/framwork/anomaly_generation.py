# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: anomaly_generation.py
# @time: 2020/03/08
import numpy as np
from scipy.sparse import csr_matrix
import operator
import random
from load_uci_messages import load_uci_messages


def anomaly_generation(ini_graph_percent, anomaly_percent, data, n, m):
    """ generate anomaly
    split the whole graph into training network which includes parts of the
    whole graph edges(with ini_graph_percent) and testing edges that includes
    a ratio of manually injected anomaly edges, here anomaly edges mean that
    they are not shown in previous graph;
     input: ini_graph_percent: percentage of edges in the whole graph will be
                                sampled in the intitial graph for embedding
                                learning
            anomaly_percent: percentage of edges in testing edges pool to be
                              manually injected anomaly edges(previous not
                              shown in the whole graph)
            data: whole graph matrix in sparse form, each row (nodeID,
                  nodeID) is one edge of the graph
            n:  number of total nodes of the whole graph
            m:  number of edges in the whole graph
     output: synthetic_test: the testing edges with injected abnormal edges,
                             each row is one edge (nodeID, nodeID, label),
                             label==0 means the edge is normal one, label ==1
                             means the edge is abnormal;
             train:  the sparse format of the training network, each row
                        (nodeID, nodeID)
    """
    # np.random.seed(1)
    print('Generating anomalous dataset...\n')
    print('Initial network edge percent: ' + str(ini_graph_percent * 100))
    print('\n')
    print('Initial anomaly percent : ' + str(anomaly_percent * 100))
    print('\n')
    train_num = int(np.floor(ini_graph_percent * m))

    # region train and test edges
    # select top train_num edges(0:train_num) as in the training set
    train = data[:train_num, :]
    train_ = np.unique(train)
    n_train = len(train_)
    adj = np.zeros((n , n))
    for edge in train:
        adj[edge[0] - 1][edge[1] - 1] = adj[edge[0] - 1][edge[1] - 1] + 1
        # adj[edge[1] - 1][edge[0] - 1] = adj[edge[1] - 1][edge[0] - 1] + 1
    # nodes=np.unique(data)

    test = data[train_num:, :]
    # test_num = test.shape[0]

    # train = data[0:train_num, :]

    # select the other edges as the testing set
    # test = data[train_num:, :]
    # endregion

    # region Read Cluster labeling from membership.txt file

    # endregion

    # region Fake Edge Generation
    # generate fake edges that are not exist in the whole graph, treat them as anamalies
    # idx_1 = np.expand_dims(np.transpose(np.random.choice(n, m)) + 1, axis=1)
    # idx_2 = np.expand_dims(np.transpose(np.random.choice(n, m)) + 1, axis=1)
    # generate_edges = np.concatenate((idx_1, idx_2), axis=1)

    # genertate abnormal edges
    # fake_edges = np.array([x for x in generate_edges if labels[x[0] - 1] != labels[x[1] - 1]])

    # remove self-loops and duplicates and order fake edges
    anomaly_num = int(np.floor(anomaly_percent * np.size(test, 0)))
    idx_test = np.zeros([np.size(test, 0) + anomaly_num, 1], dtype=np.int32)
    anomaly_pos = np.random.choice(np.size(idx_test, 0), anomaly_num, replace=False)
    idx_test[anomaly_pos] = 1



    # endregion

    # region Take anomaly_num edges from fake_edges as anomaly

    # anomalies = fake_edges[0:anomaly_num, :]
    # endregion

    # region Put anomaly edges in test_edges in random positions

    # randsample: sample without replacement
    # it's different from datasample!



    # anomaly_pos = np.random.choice(100, anomaly_num, replace=False)+200



    # endregion

    # region Prepare Synthetic test Edges
    idx_anomalies = np.nonzero(idx_test.squeeze() == 1)
    idx_normal = np.nonzero(idx_test.squeeze() == 0)
    test_aedge = np.zeros([np.size(idx_test, 0), 2], dtype=np.int32)
    test_aedge[idx_normal] = test
    # synthetic_test[idx_anomalies, 0:2] = anomalies
    test_edge = processEdges(idx_anomalies[0], test_aedge, adj)
    synthetic_test = np.concatenate((test_edge, idx_test), axis=1)


    # endregion

    # train_mat = csr_matrix((np.ones([np.size(train, 0)], dtype=np.int32), (train[:, 0] - 1, train[:, 1] - 1)),shape=(n, n))
    # sparse(train(:,1), train(:,2), ones(length(train), 1), n, n) #TODO: node addition
    # train_mat = train_mat + train_mat.transpose()

    return n_train, train, synthetic_test


def processEdges(idx_anomalies, test_aedge, adj):
    """
    remove self-loops and duplicates and order edge
    :param fake_edges: generated edge list
    :param data: orginal edge list
    :return: list of edges
    """
    # idx_fake = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] > 0)
    #
    # tmp = fake_edges[idx_fake]
    # tmp[:, [0, 1]] = tmp[:, [1, 0]]  # 调整前后顺序，使得边信息（node1，node2）中node1<=node2
    # fake_edges[idx_fake] = tmp
    for idx in idx_anomalies:
        flag = 0
        th = np.max(test_aedge[0:idx, :])
        idx_1, idx_2 = np.random.choice(th, 2, replace=False) + 1
        while (adj[idx_1 - 1][idx_2 - 1] != 0):
            idx_1, idx_2 = np.random.choice(th, 2, replace=False) + 1
        while (flag == 0):
            for edge in test_aedge[0:idx, :]:
                if (idx_1 == edge[0] and idx_2 == edge[1]) or (idx_1 == edge[1] and idx_2 == edge[0]) :
                    flag = 1
                    break
                else :
                    continue
            if flag == 0 :
                test_aedge[idx, 0] = idx_1
                test_aedge[idx, 1] = idx_2
                break
            else :
                idx_1, idx_2 = np.random.choice(th, 2, replace=False) + 1
                flag = 0
    # idx_remove_dups = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] < 0)
    #
    # fake_edges = fake_edges[idx_remove_dups]

        # a = fake_edges.tolist()
        # b = data.tolist()
        # c = []
        #
        # for i in a:
        #     if i not in b:
        #         c.append(i)
    # # fake_edges = np.array(c)
    # fake_edges = c
    # uniqueEdge=[]
    # for edge in fake_edges:
    #     if edge in uniqueEdge:
    #         continue
    #     else:
    #         uniqueEdge.append(edge)
    return test_aedge


def edgeList2Adj(data):
    """
    converting edge list to graph adjacency matrix
    :param data: edge list
    :return: adjacency matrix which is symmetric
    """

    data = tuple(map(tuple, data))

    n = max(max(user, item) for user, item in data)  # Get size of matrix
    matrix = np.zeros((n, n))
    for user, item in data:
        matrix[user - 1][item - 1] = 1  # Convert to 0-based index.
        matrix[item - 1][user - 1] = 1  # Convert to 0-based index.
    return matrix


if __name__ == "__main__":
    data_path = '../opsahl-ucsocial/out.opsahl-ucsocial'
    data, n, m = load_uci_messages(data_path, 0.3)
    edges = data
    vertices = np.unique(edges)
    train, synthetic_test = anomaly_generation(0.5, 0.8, edges, n, m)

    # print(train)
    print(train.shape)
    print('\n')
    # print(test.shape)
    # print(train_mat.todense())
    print(synthetic_test[:20,:])
    print(synthetic_test[-20:-1, :])
    print(synthetic_test.shape)
