########################
# Convenience methods

class unlocked_hists:
    """
    Context manager, for temporarily ensuring that a set of histories
    are unlocked, without changing their state permanently.
    """
    def __init__(self, *hists):
        self.hists = hists
        self.lock_states = None
    def __enter__(self):
        self.lock_states = {h: h.locked for h in self.hists}
        for h in self.hists:
            h.unlock()
    def __exit__(self, exc_type, exc_value, traceback):
        for h, locked in self.lock_states.items():
            if locked:
                h.lock(warn=False)
