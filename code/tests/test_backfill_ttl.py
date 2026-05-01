import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from train import BackfillManager

def test_ttl_expires_old_entries():
    mgr = BackfillManager(mark_present=True, ttl=3)
    t = torch.randn(8, 16)
    mgr.update(idx=0, completed_tensor=t, epoch=1)
    mgr.update(idx=1, completed_tensor=t, epoch=2)
    assert len(mgr.data) == 2
    mgr.expire(current_epoch=4)  # 1+3=4，epoch=4 时 idx=0 恰好过期
    assert 0 not in mgr.data, "idx=0 应在 epoch=4 时过期（写入 epoch=1, TTL=3）"
    assert 1 in mgr.data, "idx=1 不应过期（写入 epoch=2, TTL=3）"

def test_ttl_disabled_when_zero():
    mgr = BackfillManager(mark_present=True, ttl=0)
    t = torch.randn(4, 8)
    mgr.update(idx=5, completed_tensor=t, epoch=1)
    mgr.expire(current_epoch=1000)
    assert 5 in mgr.data, "ttl=0 时不应删除任何条目"

def test_update_stores_epoch():
    mgr = BackfillManager(mark_present=True, ttl=5)
    t = torch.randn(4, 8)
    mgr.update(idx=10, completed_tensor=t, epoch=3)
    assert 10 in mgr.data
    assert mgr.timestamps[10] == 3
