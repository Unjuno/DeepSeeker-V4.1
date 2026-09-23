"""Multi-token verify window indices (#29)."""

from __future__ import annotations

import numpy as np

from deepseeker.mlx_runner import Runner


def test_window_topk_prefill_shape():
    a = Runner._window_topk(128, 7, 0)
    assert a.shape == (7, 7)


def test_window_topk_decode_single():
    b = Runner._window_topk(128, 1, 7)
    assert b.shape == (1, 128)
    arr = np.asarray(b)[0]
    # Reference: oldest-first ring; slots > start_pos masked -1.
    # start_pos=7, win=128 → valid 0..7 at the tail after 120 leading -1s.
    assert list(arr[:120]) == [-1] * 120
    assert list(arr[120:]) == list(range(8))


def test_window_topk_multi_history_and_chunk():
    # ring size 8, already filled 0..2, chunk positions 3..6
    c = Runner._window_topk(8, 4, 3)
    assert c.shape == (4, 12)
    # query at pos 3: history 0,1,2 then chunk col win+0=8
    assert list(np.asarray(c)[0][:4]) == [0, 1, 2, 8]
    # query at pos 6: history 0,1,2 + chunk 8..11
    assert list(np.asarray(c)[3][:7]) == [0, 1, 2, 8, 9, 10, 11]


def test_window_topk_multi_ring_boundary():
    # win=4, start=5: row0 q=2,3,4 history slots 2,3,0; chunk q=5 → win+0=4
    d = Runner._window_topk(4, 3, 5)
    assert d.shape == (3, 7)
    row0 = np.asarray(d)[0]
    assert list(row0[:4]) == [2, 3, 0, 4]
    # row1 (p=6): history q=3,4 → 3,0; chunk q=5,6 → 4,5
    row1 = np.asarray(d)[1]
    assert list(row1[:4]) == [3, 0, 4, 5]
