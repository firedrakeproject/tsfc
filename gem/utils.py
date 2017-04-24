from __future__ import absolute_import, print_function, division
from six import viewitems

import collections


# This is copied from PyOP2, and it is here to be available for both
# FInAT and TSFC without depending on PyOP2.
class cached_property(object):
    """A read-only @property that is only evaluated once. The value is cached
    on the object itself rather than the function or class; this should prevent
    memory leakage."""
    def __init__(self, fget, doc=None):
        self.fget = fget
        self.__doc__ = doc or fget.__doc__
        self.__name__ = fget.__name__
        self.__module__ = fget.__module__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        obj.__dict__[self.__name__] = result = self.fget(obj)
        return result


class OrderedSet(collections.MutableSet):
    """A set that preserves ordering, useful for deterministic code
    generation."""

    def __init__(self, iterable=None):
        self._list = list()
        self._set = set()

        if iterable is not None:
            for item in iterable:
                self.add(item)

    def __contains__(self, item):
        return item in self._set

    def __iter__(self):
        return iter(self._list)

    def __len__(self):
        return len(self._list)

    def __repr__(self):
        return "OrderedSet({0})".format(self._list)

    def add(self, value):
        if value not in self._set:
            self._list.append(value)
            self._set.add(value)

    def discard(self, value):
        # O(n) time complexity: do not use this!
        if value in self._set:
            self._list.remove(value)
            self._set.discard(value)


def groupby(iterable, key=None):
    """Groups objects by their keys.

    :arg iterable: an iterable
    :arg key: key function

    :returns: list of (group key, list of group members) pairs
    """
    if key is None:
        key = lambda x: x
    groups = collections.OrderedDict()
    for elem in iterable:
        groups.setdefault(key(elem), []).append(elem)
    return viewitems(groups)


def make_proxy_class(name, cls):
    """Constructs a proxy class for a given class.  Instance attributes
    are supposed to be listed e.g. with the unset_attribute decorator,
    so that this function find them and create wrappers for them.

    :arg name: name of the new proxy class
    :arg cls: the wrapee class to create a proxy for
    """
    def __init__(self, wrapee):
        self._wrapee = wrapee

    def make_proxy_property(name):
        def getter(self):
            return getattr(self._wrapee, name)
        return property(getter)

    dct = {'__init__': __init__}
    for attr in dir(cls):
        if not attr.startswith('_'):
            dct[attr] = make_proxy_property(attr)
    return type(name, (), dct)
