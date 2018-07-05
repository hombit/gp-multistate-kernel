from __future__ import division

from collections import namedtuple, OrderedDict
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

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

    def __repr__(self):
        return 'Frozen' + repr(self._d)


_ATTRS = ('x', 'y', 'err')
StateData = namedtuple('StateData', _ATTRS)
ScikitLearnData = namedtuple('ScikitLearnData', _ATTRS + ('norm',))


class MultiStateData(object):
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

    @property
    def odict(self):
        return self._dict

    @property
    def arrays(self):
        return self._scikit

    @property
    def norm(self):
        return self._scikit.norm

    def keys(self):
        return self.odict.keys()

    @staticmethod
    def _x_2d_from_1d(x_1d_):
        return np.block([
            [np.full_like(x, i).reshape(-1,1), np.asarray(x).reshape(-1,1)] for i, x in enumerate(x_1d_)
        ]).reshape(-1, 2)

    def __mul__(self, other):
        norm = self.norm * other
        return MultiStateData.from_arrays(self.arrays.x, self.arrays.y, self.arrays.err, norm, keys=self.keys())

    def __rmul__(self, other):
        return self * other

    def __add_sub_helper(self, other, op):
        if not np.allclose(self.arrays.x, other.arrays.x):
            raise ValueError('objects should have the same x values')
        
        def generate_items():
            for key, state_data in self.odict.items():
                x = state_data.x
                y = op(state_data.y, other.odict[key].y)
                err = np.sqrt(state_data.err**2 + other.odict[key].err**2)
                yield (key, (x, y, err))
       
        return MultiStateData.from_items(generate_items())

    def __add__(self, other):
        return self.__add_sub_helper(other, op=np.add)

    def __sub__(self, other):
        return self.__add_sub_helper(other, op=np.subtract)

    def append_dict(self, d):
        """Add data from dictionary

        Parameters
        ----------
        d: dict-like
            Dictionary that is similar to `.odict`
        """
        self_keys = set(self.keys())
        d_keys = set(d.keys())

        def state_data_generator():
            for k in self_keys.intersection(d_keys):
                x = np.hstack((self.odict[k].x, d[k].x))
                y = np.hstack((self.odict[k].y, d[k].y))
                err = np.hstack((self.odict[k].err, d[k].err))
                yield k, StateData(x, y, err)
            for k in self_keys - d_keys:
                yield k, self.odict[k]
            for k in d_keys - self_keys:
                yield k, d[k]
        new_msd = self.from_state_data(state_data_generator())
        self._dict = new_msd.odict
        self._scikit = new_msd.arrays

    def append(self, other):
        """Add data from another MultiStateData object

        Parameters
        ----------
        other: MultiStateData
        """
        if not isinstance(other, MultiStateData):
            raise TypeError('Cannot append the object of type {}'.format(type(other)))
        self.append_dict(other.odict)

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
        return self._x_2d_from_1d([x]*len(self.keys()))

    def convert_arrays(self, x, y, err):
        """Get new MultiStateKernel object from scikit-learn style arrays

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
        MultiStateData
            New `MultiStateData` object with the same `.norm` and `.keys` as
            original
        """
        return self.from_arrays(x, y, err, self.norm, keys=self.keys())

    @classmethod
    def from_items(cls, items):
        """Construct from iterable of (key: (x, y, err))

        Raises
        -------
        ValueError: inconsistent input data shapes
        """
        return cls.from_state_data((k, StateData(*v)) for k,v in items)

    @staticmethod
    def __norm(y):
        return y.std() or y[0] or 1

    @classmethod
    def from_state_data(cls, *args, **kwargs):
        """Construct from iterable of (key: object), where object should has
        as attributes `x`, `y` and `err`, all are 1-D numpy.ndarray

        Raises
        -------
        ValueError: inconsistent input data shapes
        """
        d = FrozenOrderedDict(*args, **kwargs)
        for k, v in iteritems(d):
            if not len(v.x) == len(v.y) == len(v.err):
                raise ValueError('{} key has different array shapes'.format(k))
        x = cls._x_2d_from_1d((v.x for v in itervalues(d)))
        y = np.hstack((v.y for v in itervalues(d)))
        norm = MultiStateData.__norm(y)
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

        Raises
        -------
        IndexError: inconsistent input data shapes
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

        Raises
        -------
        IndexError: inconsistent input data shapes
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


__all__ = ('FrozenOrderedDict', 'StateData', 'MultiStateData',
           'data_from_items', 'data_from_state_data', 'data_from_arrays')
