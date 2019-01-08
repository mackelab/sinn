import luigi
import hashlib
from .common import Model

# =================================
# Specialized Luigi parameter types
# =================================

class ModelParameter(luigi.Parameter):
    # Because running the task may change the history, we
    # need to freeze its hash based on when it's created.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._phash = None
        self._hhash = None
    def serialize(self, model):
        if not isinstance(model, Model):
            raise TypeError("`ModelParameter` expects an argument of type "
                            "`sinn.models.Model`")
        if self._hhash is None:
            assert self._phash is None
            self._phash = self.serialize_params(model)     # Returns bytes
            self._hhash = self.serialize_histories(model)  # Returns bytes
        else:
            assert self._phash == self.serialize_params(model)
        return hashlib.md5(self._phash + self._hhash).hexdigest()
    def serialize_params(self, model):
        values = (p.get_value() if hasattr(p, 'get_value') else p for p in model.params)
        s = "".join([name + "-" + str(val) for name, val in zip(model.params._fields, values)])
        return s.encode('utf-8')
    def serialize_histories(self, model):
        return b"".join([h.digest for h in model.statehists])

class PyMCModelParameter(ModelParameter):
    def serialize_params(self, model):
        values = (p.get_value() if hasattr(p, 'get_value') else p for p in model.params)
        priors = (getattr(p, 'prior', None) for p in model.params)
        # We represent a prior by it's IPython Latex representation, because
        # that includes the variable name, distribution name and parameters
        s = "".join([name + "-" + (prior._repr_latex_() if prior is not None else str(value))
                     for name, prior, value in zip(model.params._fields, priors, values)])
        return s.encode('utf-8')
