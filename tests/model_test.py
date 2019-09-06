from sinn.histories import History, Series
from sinn.models import Model, HistSpec

class TestModel(sinn.models.Model):
    histories = {'x': HistSpec(type='any', default=Series),
                 'y': HistSpec(Series),
                 'z': Series}
    assert histories['x'].type is History
    assert histories['y'].type is Series and histories['y'].default is Series
    assert histories['z'].type is Series and histories['z'].default is Series

    # ParamSpec: shape,
    parameters = {'μ': ParamSpec()}
