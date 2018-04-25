from __future__ import division

from collections import Mapping, namedtuple, OrderedDict

import numpy as np
from six import iteritems, iterkeys, itervalues


class FrozenOrderedDict(Mapping):
    """Immutable ordered dictionary

    It is based on collections.OrderedDict, so it remembers insertion order"""

    def __init__(self, *args, **kwargs):
        self._d = OrderedDict(*args, **kwargs)
        self._hash = None

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, item):
        return self._d[item]

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(iteritems(self._d))
        return self._hash


_ATTRS = ('x', 'y', 'err')
StateData = namedtuple('StateData', _ATTRS)
ScikitLearnData = namedtuple('ScikitLearnData', _ATTRS + ('norm',))


class MultiStateData:
    """Multi state data class

    This class holds two representation of the multi state data.
    The first representation is a frozen ordered dictionary
    `.odict` composed from key - trinity of `x`, `y` `err` (all 1-D arrays).
    The second representation is `.arrays` namedtuple composed from three
    `scikit-learn` friendly arrays: `x` (2-D, as needed by
    `MultiStateKernel`), `y` and `err`, and additional constant `norm`, that
     should multiplies `y` and `err` to get `.odict` values.

    This class shouldn't be constructed by `__init__` but by class
    methods

    Parameters
    ----------
    state_data_odict: FrozenOrderedDict[str: StateData or numpy.recarray]
        Ordered dictionary of the pairs of objects with .x, .y, .err
        attributes, all of them should by 1-D numpy.ndarray
    scikit_learn_data: ScikitLearnData
        Object with .x (2-D numpy.ndarray), .y (1-D numpy.ndarray),
        .err (1-D numpy.ndarray), .norm (positive float).

    Attributes
    ----------
    odict: FrozenOrderedDict[str: StateData or numpy.recarray]
    arrays: ScikitLearnData
    norm: float
    keys: tuple

    Methods
    -------
    key(idx)
        State name by its index
    idx(key)
        State index by its name
    convert_arrays(x, y, err)
        New `MultiStateData` object from `scikit-learn` style arrays

    """

    def __init__(self, state_data_odict, scikit_learn_data):
        self._dict = state_data_odict
        self._scikit = scikit_learn_data

        self._keys_tuple = tuple(iterkeys(self._dict))
        self._state_idx_dict = {x: i for i,x in enumerate(self._dict)}

    @property
    def odict(self):
        return self._dict

    @property
    def arrays(self):
        return self._scikit

    @property
    def norm(self):
        return self._scikit.norm

    @property
    def keys(self):
        return self._keys_tuple

    def key(self, idx):
        """Get state name by index"""
        return self._keys_tuple[idx]

    def idx(self, key):
        """Get state index by name"""
        return self._state_idx_dict[key]

    @staticmethod
    def _x_2d_from_1d(x_1d_):
        return np.block(list(
            [np.full_like(x, i).reshape(-1,1), np.asarray(x).reshape(-1,1)] for i, x in enumerate(x_1d_)
        ))

    def sample(self, x):
        """Generate scikit-learn style sample from 1-d array

        Parameters
        ----------
        x: 1-D numpy.ndarray
            `x` sample data, it assumes to be the sample for every state

        Returns
        -------
        2-D numpy.ndarray
            `X`-data in the format specified by `MultiStateKernel`
        """
        return self._x_2d_from_1d([x]*len(self.keys))

    def convert_arrays(self, x, y, err):
        """Get new {class_name} object from scikit-learn style arrays

        Parameters
        ----------
        x: 2-D numpy.ndarray
            `X`-data in the format specified by `MultiStateKernel`: the first
            column is th state index, the second column is coordinate.
        y: 1-D numpy.ndarray
            `y`-data
        err: 1-D numpy.ndarray
            Errors for `y`

        Returns
        -------
        {class_name}
            New {class_name} object with the same `norm` and `keys` as
            original
        """.format(class_name=self.__name__)
        return self.from_arrays(x, y, err, self.norm, keys=self.keys)

    @classmethod
    def from_items(cls, items):
        """Construct from iterable of (key: (x, y, err))"""
        return cls.from_state_data((k, StateData(*v)) for k,v in items)

    @classmethod
    def from_state_data(cls, *args, **kwargs):
        """Construct from iterable of (key: object), where object should has
        as attributes `x`, `y` and `err`, all are 1-D numpy.ndarray
        """
        d = FrozenOrderedDict(*args, **kwargs)
        x = cls._x_2d_from_1d((v.x for v in itervalues(d)))
        y = np.hstack((v.y for v in itervalues(d)))
        if y.size == 0:
            raise ValueError('Arrays should have non-zero length')
        norm = y.std() or y[0] or 1
        y /= norm
        err = np.hstack((v.err for v in itervalues(d))) / norm
        return cls(d, ScikitLearnData(x=x, y=y, err=err, norm=norm))

    @classmethod
    def from_arrays(cls, x, y, err, norm=1, **kwargs):
        """Construct from scikit-learn style arrays

        Parameters
        ----------
        x: 2-D numpy.ndarray
            `X`-data in the format specified of `MultiStateKernel`: the first
            column is th state index, the second column is coordinate.
        y: 1-D numpy.ndarray
            `y`-data
        err: 1-D numpy.ndarray
            Errors for `y`
        norm: positive float, optional
            The positive constant to multiply `y` and `err` to obtain their
            original values
        keys: array_like, optional
            The names for states. The default is integral indexes

        """
        return cls.from_scikit_learn_data(ScikitLearnData(x=x, y=y, err=err, norm=norm), **kwargs)

    @classmethod
    def from_scikit_learn_data(cls, data, keys=None):
        """Construct from ScikitLearnData

        Parameters
        ----------
        data: ScikitLearnData
            An object with `x`, `y`, `err` and `norm` attributes. For details
            of these attributes see `.from_arrays()`
        keys: array_like, optional
            The names for states. The default is integral indexes
        """
        if keys is None:
            keys = np.unique(data.x[:,0])

        def multi_state_data_generator():
            for i, key in enumerate(keys):
                idx = data.x[:,0] == i
                yield (key, StateData(data.x[idx,1], data.y[idx]*data.norm, data.err[idx]*data.norm))
        return cls(FrozenOrderedDict(multi_state_data_generator()), data)


data_from_items = MultiStateData.from_items
data_from_state_data = MultiStateData.from_state_data
data_from_arrays = MultiStateData.from_arrays


__all__ = ('FrozenOrderedDict', 'StateData', 'data_from_items', 'data_from_state_data', 'data_from_arrays')