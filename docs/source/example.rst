Example: NN layer with OJAX
========================================

Here is a self-contained example showing how OJAX can be used to define a fully
connected layer for deep learning. The highlighted regions below showcase two
important characteristics of OJAX:

* OJAX has seamless JAX integration: use JAX transforms and functions anywhere.
  And they work as intended.
* OJAX is "pure like JAX": no in-place update. New instances with updated
  states are always returned instead.

.. literalinclude:: ../../test/test_example.py
  :language: python
  :emphasize-lines: 23,29,43-44,56-57

For a full-fledged NN library with module system, optimizers, interface with
impure codebase (e.g., dataloader and log), and fully ``jit``-able and
parallelizable high-level functions for NN training, stay tuned for
`OJAX-NN <https://github.com/ysngshn/ojax-nn>`_.