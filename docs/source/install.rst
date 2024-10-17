Installing OJAX
===============

Dependencies
------------

OJAX is written in pure Python and requires Python 3.10+ (same as JAX).

It only depends on JAX, which can be installed following the
`JAX official installation guide`_.

Since JAX is still constantly changing the API, OJAX releases will align with
the API of the concurrent latest stable JAX releases.

Installing OJAX
---------------

Once JAX is installed, OJAX can be easily installed with ``pip``:

.. code-block:: bash

  pip install ojax

.. _JAX official installation guide: https://jax.readthedocs.io/en/latest/installation.html