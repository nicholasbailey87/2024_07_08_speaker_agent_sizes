import torch
import scipy.stats

def lech_dist(A, B):
    """
    given two tensors A, and B, with the item index along the first dimension,
    and each tensor is 2-dimensional, this will calculate the lechenstein distance
    between each pair of examples between A and B
    """
    # print('')
    N_a = A.size(0)
    N_b = B.size(0)
    E = A.size(1)
    assert E == B.size(1)
    assert len(A.size()) == 2
    assert len(B.size()) == 2
    A = A.unsqueeze(1).expand(N_a, N_b, E)
    B = B.unsqueeze(0).expand(N_a, N_b, E)
    AeqB = A == B
    dists = AeqB.sum(dim=-1)
    dists = dists.float() / E
    return dists

def tri_to_vec(tri):
    """
    returns lower triangle of a square matrix, as a vector, excluding the diagonal

    eg given

    1 3 9
    4 3 7
    2 1 5

    returns:

    4 2 1
    """
    assert len(tri.size()) == 2
    assert tri.size(0) == tri.size(1)
    K = tri.size(0)
    res_size = (K - 1) * K // 2
    tri_cpu = tri.cpu()
    res = torch.zeros(res_size, dtype=tri.dtype)
    pos = 0
    for k in range(K - 1):
        res[pos:pos + (K - k - 1)] = tri[k + 1:, k]
        pos += (K - k - 1)
    return res

def calc_squared_euc_dist(one, two):
    """
    input: two arrays, [N1][E]
                       [N2][E]
    output: one matrix: [N1][N2]
    """
    one_squared = (one * one).sum(dim=1)
    two_squared = (two * two).sum(dim=1)
    transpose = one @ two.transpose(0, 1)
    squared_dist = one_squared.unsqueeze(1) + two_squared.unsqueeze(0) - 2 * transpose
    return squared_dist

def topographic_similarity(utts, labels):
    """
    (quoting Angeliki 2018)
    "The intuition behind this measure is that semantically similar objects should have similar messages."

    a and b should be discrete; 2-dimensional. with item index along first dimension, and attribute index
    along second dimension
    """
    utts_pairwise_dist = tri_to_vec(lech_dist(utts, utts))
    labels_pairwise_dist = tri_to_vec(lech_dist(labels, labels))
    # max_diff = (utts_pairwise_dist - utts_pairwise_dist[0]).abs().max().item()
    # if max_diff == 0:
    #     print('utts all identical => forcing rho to zero')
    #     return 0
    rho, _ = scipy.stats.spearmanr(a=utts_pairwise_dist, b=labels_pairwise_dist)
    if rho != rho:
        # if rho is nan, we'll assume taht utts was all the same value. hence rho
        # is zero. (if labels was all the same value too, rho would be unclear, but
        # since the labels are provided by the dataset, we'll assume that they are diverse)
        max_utts_diff = (utts_pairwise_dist - utts_pairwise_dist[0]).abs().max().item()
        max_labels_diff = (labels_pairwise_dist - labels_pairwise_dist[0]).abs().max().item()
        print('rho is zero, max_utts_diff', max_utts_diff, 'max_labels_diff', max_labels_diff)
        rho = 0
    return rho

def uniqueness(a):
    """
    given 2 dimensional discrete tensor a, will count the number of unique vectors,
    and divide by the total number of vectors, ie returns the fraction of vectors
    which are unique
    """
    v = set()
    N, K = a.size()
    for n in range(N):
        v.add(','.join([str(x) for x in a[n].tolist()]))
    return (len(v) - 1) / (N - 1)   # subtract 1, because if everything is identical, there would still be 1
