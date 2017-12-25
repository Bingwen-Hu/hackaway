doc_test module
===============

.. automodule:: doc_test
:members:
        :undoc-members:
                :show-inheritance:

            Examples:

.. testsetup::

    from doc_test import square

.. doctest::

    >>> square(100)

    >>> square(0)
    0
    >>> square(1)
    1
    >>> square(3)
    9
    >>> square()
    Traceback (most recent call last):
    ...
    TypeError: square() missing 1 required positional argument: 'n'
    >>> square('x')
    Traceback (most recent call last):
    ...
    TypeError: can't multiply sequence by non-int of type 'str'
