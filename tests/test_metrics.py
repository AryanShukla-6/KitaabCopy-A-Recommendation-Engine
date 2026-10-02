import math

from kitaabcopy.evaluate import dcg_at_k, ndcg_at_k, precision_at_k, recall_at_k


def test_precision_at_k():
    assert precision_at_k([3, 0, 2], 3) == 2 / 3
    assert precision_at_k([0, 1, 1], 3) == 0.0          # label 1 is below the relevance threshold


def test_recall_at_k():
    assert recall_at_k([3, 0, 2], n_relevant=4, k=3) == 0.5
    assert math.isnan(recall_at_k([0, 0, 0], n_relevant=0, k=3))


def test_ndcg_perfect_and_worst():
    all_labels = [3, 2, 1, 0]
    assert abs(ndcg_at_k([3, 2, 1], all_labels, 3) - 1.0) < 1e-9
    assert ndcg_at_k([0, 0, 0], all_labels, 3) == 0.0
    # hand-computed: DCG = 7 + 3/log2(3) + 0 ; ideal = 7 + 3/log2(3) + 1/2
    got = ndcg_at_k([3, 2, 0], all_labels, 3)
    want = (7 + 3 / math.log2(3)) / (7 + 3 / math.log2(3) + 0.5)
    assert abs(got - want) < 1e-9


def test_dcg_order_matters():
    assert dcg_at_k([3, 0], 2) > dcg_at_k([0, 3], 2)
