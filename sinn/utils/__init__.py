########################
# Convenience methods

class unlocked_hists:
    """
    Context manager, for temporarily ensuring that a set of histories
    are unlocked, without changing their state permanently.
    """
    def __init__(self, *hists, if_pending_updates="raise"):
        self.hists = hists
        self.if_pending_updates = if_pending_updates
        self.lock_states = None
    def __enter__(self):
        self.lock_states = {h: h.locked for h in self.hists}
        for h in self.hists:
            h.unlock(if_pending_updates=self.if_pending_updates)
    def __exit__(self, exc_type, exc_value, traceback):
        for h, locked in self.lock_states.items():
            if locked:
                h.lock(warn=False)
