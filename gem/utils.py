from __future__ import absolute_import, print_function, division

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
