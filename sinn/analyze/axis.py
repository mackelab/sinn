from collections import namedtuple, Callable
from mackelab_toolbox.parameters import Transform
import numpy as np

ParameterLabel = namedtuple('ParameterLabel', ['name', 'idx'])

class Axis:
    formats = ['centers', 'edges']

    def __init__(self, label, transformed_label=None, label_idx=None, *args,
                 stops=None, transformed_stops=None, format='centers',
                 transform_fn=None, inverse_transform_fn=None):
        """
        Must specify exactly one of `stops`, `transformed_stops`.
        Alternatively, can also provide a transform description as first argument
        (same format as 'parameters.TransformedVar') along with one of the stops
        arguments.

        TODO: Implement slicing

        Parameters
        ----------
        label: str or ParameterLabel

        transformed_label: str or ParameterLabel
            If only `label` is provided, defaults to "f(label)".

        label_idx: int, tuple or 'None'

        one of:
          stops: ndarray
          transformed_stops: ndarray

        format: str

        transform_fn: callable or str (Transform description)

        inverse_transform_fn: callable or str (Transform description)


        Parameters (from TransformVar description)
        ---------------------------------------
        label: ParameterSet
            Transform description

        label_idx: int, tuple or 'None'

        one of:
          stops: ndarray
          transformed_stops: ndarray

        format: str

        """
        if not( (stops is None) != (transformed_stops is None) ):  #xor
            raise ValueError("Exactly one of `stops`, `transformed_stops` must be specified.")

        if isinstance(format, Callable):
            format = format()  # In case we pass the format method rather than its evaluation
        if format not in self.formats:
            raise ValueError("`format` must be one of {}. It is '{}'."
                             .format(', '.join(["'"+f+"'" for f in self.formats]),
                                     format))
        else:
            self._format_str = format

        if ( isinstance(label, ParameterLabel)
             or isinstance(transformed_label, ParameterLabel) ):
            if label_idx is not None:
                raise ValueError("The label index is already given by a "
                                 "ParameterLabel; specifying `label_idx` is ambiguous.")
            else:
                if isinstance(label, ParameterLabel):
                    label_idx = label.idx
                    if (isinstance(transformed_label, ParameterLabel)
                        and label_idx != transformed_label.idx):
                        label_idx_t = (
                          tuple(label_idx) if isinstance(label_idx, Iterable)
                          else label_idx)
                        trans_label_idx_t = (
                          tuple(transformed_label.idx) if isinstance(transformed_label.idx, Iterable)
                          else transformed_label.idx)
                        if label_idx_t != trans_label_idx_t:
                            raise ValueError("Index of `label` and `transformed_label` "
                                             "don't match.")
                else:
                    label_idx = transformed_label.idx

        if isinstance(label, str):
            label = ParameterLabel(label, label_idx)
            if transformed_label is None:
                if transform_fn is None:
                    transformed_label = label.name
                else:
                    transformed_label = "f(" + label.name + ")"
        if isinstance(transformed_label, str):
            transformed_label = ParameterLabel(transformed_label, label_idx)

        if isinstance(label, ParameterLabel):
            # At this point string labels have be converted to ParameterLabel
            if (transform_fn is None) != (inverse_transform_fn is None):
                raise ValueError("If a transform function is specified, its "
                                 "inverse must be as well.")
            self.label = label
            self.transformed_label = transformed_label
            if transform_fn is None:
                self.to = self.back = Transform("x -> x")
            else:
                self.to = Transform(transform_fn)
                self.back = Transform(inverse_transform_fn)
            if stops is None:
                #self.transformed_stops = transformed_stops
                self.stops = self.back(transformed_stops)
            else:
                self.stops = stops
                #self.transformed_stops = self.to(stops)

        elif isinstance(label, ParameterSet):
            desc = label # More meaningful name
            if ( transform_fn is not None or inverse_transform_fn is not None):
                raise ValueError("Specifying a transform function along with a "
                                 "transform description is ambiguous.")
            if transformed_label is not None:
                raise ValueError("Specifying a transformed label along with a "
                                 "transform description is ambiguous.")
            assert(set(['name', 'to', 'back']).issubset(set(desc.keys())))

            label, transformed_label = [lbl.strip()
                                        for lbl in desc.name.split('->')]
            self.__init__(label, transformed_label, label_idx,
                          stops=stops, transformed_stops=transformed_stops,
                          transform_fn=Transform(desc.to),
                          inverse_transform_fn=Transform(desc.back))

    def __len__(self):
        return len(self.stops)
    def __str__(self):
        return self.name + ' ({:.3}:{:.3}:{.3})'.format(a.stops[0], a.stops[1],
                                                        (a.stops[-1]-a.stops[0])/len(self))


    @property
    def transformed(self):
        """Return the transformed axis."""
        return TransformedAxis(label  = self.transformed_label,
                               untransformed_label = self.label,
                               stops  = self.to(self.stops),
                               format = self.format,
                               invert_fn = self.back,
                               revert_fn = self.to)

    @property
    def transformed_stops(self):
        return self.to(self.stops)

    @property
    def name(self):
        """Synonym for `label.name`."""
        return self.label.name
    @property
    def idx(self):
        return self.label.idx

    @property
    def start(self):
        """
        Return the start of the axis. Can be used to set limits on a plot.
        """
        return self.edges.stops[0]

    @property
    def end(self):
        """
        Return the end of the axis. Can be used to set limits on a plot.
        """
        return self.edges.stops[-1]

    @property
    def limits(self):
        """
        Return a (start, end) giving the bounds of the axis. Can be used to "
        "set limits on a plot.
        """
        return (self.start, self.end)

    @property
    def widths(self):
        """
        Return an ndarray of same length as `centers` giving each bin's width.
        """
        edges = self.edges.stops
        return abs(edges[1:] - edges[:-1])

    def format(self, format_str=None):
        if format_str is None:
            return self._format_str
        else:
            if format_str == 'current':
                return self
            elif format_str == 'edges':
                return self.edges
            elif format_str in ['centers', 'centres']:
                return self.centers
            else:
                raise ValueError("Unrecognized axis format '{}'."
                                 .format(format_str))

    @property
    def edges(self):
        """
        Return an Axis instance where stops correspond to bin edges.
        If this is already this axis' format, it is simply returned;
        otherwise, it is converted. E.g. if the current format is 'centers',
        the returned axis will have stops such that produced bins are
        centered around the current stops in **transformed** space.
        """
        if self._format_str == 'edges':
            return self
        elif self._format_str in ['centers', 'centres']:
            stops = self.transformed_stops
            dxs = (stops[1:]-stops[:-1])/2
            newstops = np.concatenate(((stops[0]-dxs[0],),
                                       (stops[1:] - dxs),
                                       (stops[-1] + dxs[-1],)))
            return Axis(self.label, self.transformed_label,
                        transformed_stops=newstops, format='edges',
                        transform_fn=self.to, inverse_transform_fn=self.back)
        else:
            raise RuntimeError("Unrecognized axis format '{}'."
                               .format(self._format_str))

    @property
    def centers(self):
        """
        Return an Axis instance where stops correspond to bin centres.
        If this is already this axis' format, it is simply returned;
        otherwise, it is converted. E.g. if the current format is 'edges',
        the returned axis will have stops at the center of each bin
        in **transformed** space.
        """
        if self._format_str in ['centers', 'centres']:
            return self
        elif self._format_str == 'edges':
            stops = self.transformed_stops
            newstops = (stops[1:]+stops[:-1])/2
            return Axis(self.label, self.transformed_label,
                        transformed_stops=newstops, format='centers',
                        transform_fn=self.to, inverse_transform_fn=self.back)
        else:
            raise RuntimeError("Unrecognized axis format '{}'."
                               .format(self._format_str))

class TransformedAxis(Axis):

    def __init__(self, label, untransformed_label, label_idx=None, *args,
                 stops=None, untransformed_stops=None, format='centers',
                 invert_fn=None, revert_fn=None):
        super().__init__(label=label, transformed_label=untransformed_label,
                         label_idx=label_idx, stops=stops, format=format,
                         transform_fn=invert_fn,
                         inverse_transform_fn=revert_fn)

        # Internal functions (like edges(), centers) require `to` and `back` to be identities
        self.invert = self.to   # transformed -> untransformed
        self.revert = self.back # untransformed -> transformed
        self.to = self.back = lambda x: x
        # Change names of 'transformed' properties
        self.untransformed_label = self.transformed_label
        del self.transformed_label

    # Deactivate methods that lost their meaning
    transformed = None
    transformed_stops = None

    @property
    def untransformed(self):
        return Axis(label = self.untransformed_label,
                    transformed_label = self.label,
                    stops = self.back(self.stops),
                    format = self.format,
                    transform_fn = self.to,
                    inverse_transform_fn = self.back)

    @property
    def untransformed_stops(self):
        return self.back(self.stops)
