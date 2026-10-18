"""Restore epoch-boundary repair training without resetting optimizer or plateau state."""
def restore_training(saved, repairs, optimizer, scheduler, contract):
    previous = dict(saved['contract'])
    current = dict(contract)
    old_limit = previous.pop('epochs_max')
    new_limit = current.pop('epochs_max')
    assert previous == current, 'Resume cannot change the training/data/model contract'
    assert new_limit > saved['epoch'] and new_limit >= old_limit
    history = saved['history']
    assert [r['epoch'] for r in history] == list(range(1, saved['epoch'] + 1))
    assert saved['best'] == max(r['ap70'] for r in history)
    assert saved['stale'] == history[-1]['stale_epochs']
    repairs.load_state_dict(saved['repairs'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    assert optimizer.param_groups[0]['lr'] == history[-1]['lr']
    assert scheduler.last_epoch == saved['epoch']
    return saved['epoch'] + 1, saved['best'], saved['significant'], saved['stale'], list(history)
