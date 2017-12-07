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
    return groups.items()


def make_proxy_class(name, cls):
    """Constructs a proxy class for a given class.

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


# Implementation of dynamically scoped variables in Python.
class UnsetVariableError(LookupError):
    pass


_unset = object()


class DynamicallyScoped(object):
    """A dynamically scoped variable."""

    def __init__(self, default_value=_unset):
        if default_value is _unset:
            self._head = None
        else:
            self._head = (default_value, None)

    def let(self, value):
        return _LetBlock(self, value)

    @property
    def value(self):
        if self._head is None:
            raise UnsetVariableError("Dynamically scoped variable not set.")
        result, tail = self._head
        return result


class _LetBlock(object):
    """Context manager representing a dynamic scope."""

    def __init__(self, variable, value):
        self.variable = variable
        self.value = value
        self.state = None

    def __enter__(self):
        assert self.state is None
        value = self.value
        tail = self.variable._head
        scope = (value, tail)
        self.variable._head = scope
        self.state = scope

    def __exit__(self, exc_type, exc_value, traceback):
        variable = self.variable
        assert self.state is variable._head
        value, variable._head = variable._head
        self.state = None
