



OJAX
====

::

    +-------------------------------------+
    |       __              __      __    |
    |      /_/\            / /\    /_/\   |
    |     /_/  \          / /  \  /_/ /   |
    |    /_/ /\ \        / / /\ \/_/ /    |
    |   /_/ /\_\ \      / / /\_\ \/ /     |
    |  /_/ /  \_\ \    / / /__\_\  /      |
    |  \_\ \  /_/ /   / / ______   \      |
    |   \_\ \/_/ /   / / /   /_/ /\ \     |
    |    \_\ \/_/___/ / /   /_/ /\_\ \    |
    |     \_\/_______/ /   /_/ /  \_\ \   |
    |      \_\_\_\_\_\/    \_\/    \_\/   |
    |                                     |
    +-------------------------------------+


`Github <https://github.com/ysngshn/ojax>`_ | `Documentation <https://ysngshn.github.io/ojax>`_

What is OJAX
------------

OJAX is a small extension of `JAX`_ to facilitate modular coding.

You might have already noticed, due to its functional nature, JAX does not pair
well with the generic Python ``class`` structure. People tend to instead write
closures/functionals which are functions that return JAX functions (e.g. the
`Stax NN library`_ from the JAX codebase), which is far from ideal for complex
projects.

OJAX lets you write JAX code using ``class`` again, with full JAX compatibility,
max flexibility, and zero worry. Here is an example custom class using OJAX
that can be directly ``jax.jit``\ ted:

.. code-block:: python

    class AddWith(ojax.OTree):
        value: float

        def __call__(self, t: jax.Array) -> jax.Array:
            return t + self.value


    add42_jitted = jax.jit(AddWith(42.0))
    print(add42_jitted(jax.numpy.ones(1)))  # [43.]

OJAX is a simple library that needs less than 1 hour to learn, and will save
you countless hours for your JAX projects!

Why OJAX
--------

::

  "Library XXX already did something similar, why reinvent the wheel?"

The short answer is: because the wheel is rounder this time ;)

Motivated by deep learning applications, there are many JAX libraries that
already propose some kind of module system: `Flax`_, `Equinox`_, `Haiku`_,
`Simple Pytree`_, `Treeo`_ / `Treex`_, `PAX`_, just to name a few.

However, none of them offers a perfect "JAX base class" that fulfills all of
the desiderata below:

* Simple to understand and use
* Flexible custom classes for general JAX computation
* Compatible with JAX and its functional paradigm

OJAX strives to define how a JAX base class should be. It provides a natural 
way to structure custom JAX code and discourages users from common pitfalls.

P.S.: the name "OJAX" is a chapeau-bas to `OCaml <https://ocaml.org>`_, an
awesome functional programming language.

How to code with OJAX
---------------------

OJAX is easy to install `following the installation guide`_.

You can have a look at the `quickstart section`_ to get
started, and there is also a simple `example code using OJAX`_.

Of course, check out the `OJAX API reference`_ for exact
definitions.

..
  links
.. _Equinox: https://github.com/patrick-kidger/equinox
.. _Flax: https://github.com/google/flax
.. _Haiku: https://github.com/deepmind/dm-haiku
.. _InitVar: https://docs.python.org/3/library/dataclasses.html#init-only-variables
.. _JAX: https://jax.readthedocs.io
.. _OJAX API reference: https://ysngshn.github.io/ojax/modules.html
.. _PAX: https://github.com/NTT123/pax
.. _Stax NN library: https://github.com/google/jax/blob/main/jax/example_libraries/stax.py
.. _Simple Pytree: https://github.com/cgarciae/simple-pytree
.. _Treeo: https://github.com/cgarciae/treeo
.. _Treex: https://github.com/cgarciae/treex
.. _example code using OJAX: https://ysngshn.github.io/ojax/example.html
.. _following the installation guide: https://ysngshn.github.io/ojax/install.html
.. _quickstart section: https://ysngshn.github.io/ojax/quickstart.html

