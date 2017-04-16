import numpy as np
import scipy.sparse as sparse


def refine2dtri(V, E, marked_elements=None):
    """
    marked_elements : array
        list of marked elements for refinement.  None means uniform.
    """
    Nel = E.shape[0]
    Nv = V.shape[0]

    if marked_elements is None:
        marked_elements = np.arange(0, Nel)

    marked_elements = np.ravel(marked_elements)

    # construct vertex to vertex graph
    col = E.ravel()
    row = np.kron(np.arange(0, Nel), [1, 1, 1])
    data = np.ones((Nel*3,))
    V2V = sparse.coo_matrix((data, (row, col)), shape=(Nel, Nv))
    V2V = V2V.T * V2V

    # compute interior edges list
    V2V.data = np.ones(V2V.data.shape)
    V2Vupper = sparse.triu(V2V, 1).tocoo()

    # construct EdgeList from V2V
    Nedges = len(V2Vupper.data)
    V2Vupper.data = np.arange(0, Nedges)
    EdgeList = np.vstack((V2Vupper.row, V2Vupper.col)).T
    Nedges = EdgeList.shape[0]

    # elements to edge list
    V2Vupper = V2Vupper.tocsr()
    edges = np.vstack((E[:, [0, 1]],
                       E[:, [1, 2]],
                       E[:, [2, 0]]))
    edges.sort(axis=1)
    ElementToEdge = V2Vupper[edges[:, 0], edges[:, 1]].reshape((3, Nel)).T

    marked_edges = np.zeros((Nedges,), dtype=bool)
    marked_edges[ElementToEdge[marked_elements, :].ravel()] = True

    # mark 3-2-1 triangles
    nsplit = len(np.where(marked_edges == 1)[0])
    edge_num = marked_edges[ElementToEdge].sum(axis=1)
    edges3 = np.where(edge_num >= 2)[0]
    marked_edges[ElementToEdge[edges3, :]] = True  # marked 3rd edge
    nsplit = len(np.where(marked_edges == 1)[0])

    edges1 = np.where(edge_num == 1)[0]
    # edges1 = edge_num[id]             # all 2 or 3 edge elements

    # new nodes (only edges3 elements)

    x_new = 0.5*(V[EdgeList[marked_edges, 0], 0]) \
        + 0.5*(V[EdgeList[marked_edges, 1], 0])
    y_new = 0.5*(V[EdgeList[marked_edges, 0], 1]) \
        + 0.5*(V[EdgeList[marked_edges, 1], 1])

    V_new = np.vstack((x_new, y_new)).T
    V = np.vstack((V, V_new))
    # indices of the new nodes
    new_id = np.zeros((Nedges,), dtype=int)
    # print(len(np.where(marked_edges == 1)[0]))
    # print(nsplit)
    new_id[marked_edges] = Nv + np.arange(0, nsplit)
    # New tri's in the case of refining 3 edges
    # example, 1 element
    #                n2
    #               / |
    #             /   |
    #           /     |
    #        n5-------n4
    #       / \      /|
    #     /    \    / |
    #   /       \  /  |
    # n0 --------n3-- n1
    ids = np.ones((Nel,), dtype=bool)
    ids[edges3] = False
    ids[edges1] = False

    E_new = np.delete(E, marked_elements, axis=0)  # E[id2, :]
    n0 = E[edges3, 0]
    n1 = E[edges3, 1]
    n2 = E[edges3, 2]
    n3 = new_id[ElementToEdge[edges3, 0]].ravel()
    n4 = new_id[ElementToEdge[edges3, 1]].ravel()
    n5 = new_id[ElementToEdge[edges3, 2]].ravel()

    t1 = np.vstack((n0, n3, n5)).T
    t2 = np.vstack((n3, n1, n4)).T
    t3 = np.vstack((n4, n2, n5)).T
    t4 = np.vstack((n3, n4, n5)).T

    E_new = np.vstack((E_new, t1, t2, t3, t4))
    return V, E_new
