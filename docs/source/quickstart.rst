OJAX Quickstart
===============


The core component of OJAX is the :py:meth:`ojax.OTree` class, which represents
an immutable `PyTree`_ and uses Python dataclass field declaration. It is
implemented using `frozen dataclass`_ which is a standard Python feature, and
it serves as a base class for all custom JAX-compatible classes.

Declaring annotated fields
--------------------------

Unlike standard Python ``class`` where fields are dynamically added via
``self.field_name = value``, OTree adopts the annotated field syntax from
Python `dataclass`_ and expects you to declare fields with type annotation.
This is more in line with the functional paradigm employed by JAX.

Here is an example (code excerpt from the :doc:`full example <example>`):

.. literalinclude:: ../../tests/test_example.py
  :language: python
  :lines: 9-13

Annotated fields can have the following patterns:

* ``field_name: type``
* ``field_name: type = default_value``
* ``field_name: type = dataclasses.field()`` with optional `dataclasses.field`_
  arguments such as ``default`` / ``default_factory`` and ``init``
* ``field_name: type =``:meth:`ojax.child` / :meth:`ojax.aux` /
  :meth:`ojax.ignore` /
  :meth:`ojax.alien` with the same optional arguments as `dataclasses.field`_

.. note::

  Type annotation is required because Python dataclass uses it to
  identify fields. Attributes without type annotation are ignored by dataclass
  and become class variables instead of dataclass fields. OTree raises a
  warning for you in this case so don't worry about accidentally making class
  variables instead of fields :)

  If you want to create a class variable, declare it explicitly with
  :obj:`typing.ClassVar` so you won't be nagged by this warning.

.. note::

  Doing accurate type annotation is helpful but not required, since it is not
  checked by Python. This said, typing can help OTree to infer how to handle
  the fields as a PyTree. No need to think too hard though, declaring
  everything as :obj:`typing.Any` is also fine, and the PyTree handling can be
  specified explicitly in any case.

In the last pattern, :meth:`ojax.child` / :meth:`ojax.aux` /
:meth:`ojax.ignore` / :meth:`ojax.ignore` are variants of
:obj:`dataclasses.field` that also specifies how an OTree should handle a field
as a PyTree. Let's discuss this point further.

Field types for OTree
---------------------

`PyTree`_ is the data structure used by JAX to operate on data collections. It
is composed of a definition part and a content part, and JAX operations act on
the content part. OTree is registered as a PyTree, and thus should decide on
how to handle its data fields. For this, fields in OTree are partitioned into
four field types:

* Auxiliary fields
    These are the fields that will be part of the PyTree definition. They are
    supposed to be static metadata that describe the characteristics of the
    OTree. They can be explicitly declared with :meth:`ojax.aux`.
* Children fields
    These are the numerical content of the OTree which is the subject of JAX
    operations. They are typically JAX arrays and sub-PyTrees and can be marked
    explicitly with :meth:`ojax.child`.
* Ignored fields
    These are dataclass fields that are omitted by the PyTree. They are
    declared with :meth:`ojax.ignore`.
* Alien fields *(since 3.1)*
    These are dataclass fields that are incompatible with PyTree
    flattening. :class:`ojax.AlienException` is raised when flattening is
    attempted on an Alien field holding a value that is not `None`. They are
    declared with :meth:`ojax.alien`.

.. warning::

  Ignored fields are not preserved after the :obj:`jax.tree.flatten` then
  :obj:`jax.tree.unflatten` transformations. Since this combo is used by common
  JAX operations to handle PyTrees, ignored fields will easily get lost. Users
  should stick with auxiliary and children fields by default, or use alien
  fields to declare incompatible fields.

For fields without explicit field type declaration, OTree infers the field type
based on the field annotation: subclasses of :class:`jax.Array` and
:class:`ojax.OTree` are assumed to be child nodes, while the rest are assumed
to be aux nodes. This inference logic is specified in the
:meth:`ojax.OTree.__infer_otree_field_type__` method and can be overridden by
subclasses.

.. warning::

  Non-OTree PyTrees such as lists of JAX arrays are not automatically detected
  as child nodes. The current inference logic only conservatively tackles the
  obvious case for your convenience. You need to explicitly handle child node
  declarations for more complex cases.

The ``__init__`` method
-------------------------

After the declaration comes the instantiation part. The following
code segment from the :doc:`full example <example>` shows an example
``__init__`` method.

.. literalinclude:: ../../tests/test_example.py
  :language: python
  :lines: 15-19

One thing to note is that the usual ``self.field = value`` assignment pattern
is no longer possible for OTree, as it is a frozen dataclass. Instead, OTree
offers an ``.assign_`` method to achieve this (inherited from the
:meth:`ojax.PureClass.assign_` method, where :class:`ojax.PureClass` defines an
immutable dataclass).

.. note::

  Standard dataclasses allows for automatic generation of ``__init__`` method
  one can use the :obj:`dataclasses.__post_init__` method and `InitVar`_ to
  customize the initialization process of OTrees. While this might be conveient
  in simple cases, it requires more expertise with the Python dataclass
  structure and can be problematic especially with class inheritance (e.g.,
  [1]_). Moreover, it will silently prevent the inheritance ``__init__`` method
  from parent classes with an automatic generation which is error-prone. Thus
  OJAX has disabled the automatic ``__init__`` method generation feature 
  *(since 3.1)*.

Updating the fields
-------------------

During JAX computations, it is sometimes desirable to update the numerical
fields of OTrees. To achieve this, OTree provides the :meth:`ojax.OTree.update`
method, which returns an updated OTree instance with the specified new
numerical data for the children fields. The following code segment from the
:doc:`full example <example>` illustrates how it is used:

.. literalinclude:: ../../tests/test_example.py
  :language: python
  :lines: 9

.. literalinclude:: ../../tests/test_example.py
  :language: python
  :lines: 26-29
  :emphasize-lines: 4

:meth:`ojax.OTree.update` preserves the PyTree structure and disallows updating
auxiliary fields. This is usually the intended behavior since JAX operations
don't alter the PyTree structure either. It is also required for some arguments
in JAX functions such as :obj:`jax.lax.scan`.

.. note::

  ``None`` is a special empty PyTree container. Thus updating numerical fields
  to ``None`` or vice versa will change the PyTree structure and trigger an
  error from the ``.update()`` method.

In case you need to create a derived OTree with different auxiliary data or
PyTree structure, the :meth:`ojax.new` method should be used instead.

.. note::

  Again, Python dataclass offers the :obj:`dataclasses.replace` method for
  updating dataclasses fields. However, it also does not preserve the PyTree
  structure, similar to :meth:`ojax.new`. Furthermore, it relies on re-calling
  the ``__init__`` method to generate the new instance and can be problematic
  for OTrees with custom ``__init__`` method. :meth:`ojax.OTree.update` and
  :meth:`ojax.new` rely on shallow copy instead and don't have such issue.

.. warning::

  The ``.assign_`` method introduced previously should not be used for updating
  the fields of OTrees. It is a low-level in-place operation that should only
  be used within the initialization functions. Otherwise the immutable paradigm
  is easily violated, potentially creating issues and subtle bugs for JAX code.

Adding JAX methods
------------------

Of course, you are free to add custom methods just like how it is done for
vanilla Python classes. Here is an example (another excerpt from the
:doc:`full example <example>`):

.. literalinclude:: ../../tests/test_example.py
  :language: python
  :lines: 9

.. literalinclude:: ../../tests/test_example.py
  :language: python
  :lines: 22-23



As an OTree is also a PyTree that JAX operations can handle, all its methods
(which have ``self`` as their first argument) can directly work with JAX
transforms including :obj:`jax.jit`.

While there is no easy way to enforce it in Python, the OTree methods should be
`pure JAX functions`_ so as to avoid potential issues when working with JAX.

General coding tips
-------------------

To avoid `"bad surprises"`_ when coding with JAX / OJAX, always follow the two
principles below:

* Data should be persistent: no in-place operations, data should be immutable.
* Functions should be pure: no side-effect, a function with the same argument
  should always return the same result and without other effects.

These principles are the basis of functional programming and are assumed by
JAX, however it is not enforced in Python as an OOP language.

Additionally, to make your code ``jax.jit`` friendly, be mindful of control
flows (e.g., ``if``, ``for``, ``while``, etc.) which might crash or slow down
the compilation, and consider using `JAX structured control primitives`_.

``OJAX`` helps you to fulfill these principles: it is designed to strongly
encourage persistent data style and its codebase is ``jax.jit`` friendly.

..
  links

.. _"bad surprises": https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
.. _InitVar: https://docs.python.org/3/library/dataclasses.html#init-only-variables
.. _PyTree: https://jax.readthedocs.io/en/latest/pytrees.html
.. _dataclass: https://docs.python.org/3/library/dataclasses.html
.. _dataclasses.field: https://docs.python.org/3/library/dataclasses.html#dataclasses.field
.. _frozen dataclass: https://docs.python.org/3/library/dataclasses.html#frozen-instances
.. _pure JAX functions: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions
.. _JAX structured control primitives: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#structured-control-flow-primitives

..
  footnotes

.. [1] When inheriting a base class where some fields have default values, if
  the current class has any field without default value, a ``TypeError`` will
  be raised. Quoting
  `PEP-557 <https://peps.python.org/pep-0557/#specification>`_: "TypeError will
  be raised if a field without a default value follows a field with a default
  value. This is true either when this occurs in a single class, or as a result
  of class inheritance."
