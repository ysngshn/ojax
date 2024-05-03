from typing import (
    TypeVar,
    Optional,
    Sequence,
)
import enum
import dataclasses
import jax
from . import pureclass


@enum.unique
class FieldType(enum.StrEnum):
    """Choices for OTree field types."""
    
    AUX = "aux"
    CHILD = "child"
    IGNORE = "ignore"


def get_field_type(f: dataclasses.Field) -> Optional[FieldType]:
    """Retrieve the OJAX field type from a ``dataclasses.Field`` object.

    Args:
        f: a field of ``ojax.OTree``.

    Returns:
        The ``ojax.FieldType`` if available, and ``None`` otherwise.
    """

    return f.metadata.get("ojax-field-type")


def aux(
    *,
    default=dataclasses.MISSING,
    default_factory=dataclasses.MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
) -> dataclasses.Field:
    """Declares an OTree field that holds auxiliary data as part of
    ``PyTreeDef``.

    This function has identical arguments and return as the
    :obj:`dataclasses.field` function.
    """

    metadata = {} if metadata is None else metadata
    metadata = {"ojax-field-type": FieldType.AUX, **metadata}
    kwargs = {
        "default": default,
        "default_factory": default_factory,
        "init": init,
        "repr": repr,
        "hash": hash,
        "compare": compare,
        "metadata": metadata,
    }
    return dataclasses.field(**kwargs)


# OTree field which holds a child PyTree node
def child(
    *,
    default=dataclasses.MISSING,
    default_factory=dataclasses.MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
):
    """Declares an OTree field that holds a child PyTree node.

    This function has identical arguments and return as the
    :obj:`dataclasses.field` function.
    """

    metadata = {} if metadata is None else metadata
    metadata = {"ojax-field-type": FieldType.CHILD, **metadata}
    kwargs = {
        "default": default,
        "default_factory": default_factory,
        "init": init,
        "repr": repr,
        "hash": hash,
        "compare": compare,
        "metadata": metadata,
    }
    return dataclasses.field(**kwargs)


def ignore(
    *,
    default=dataclasses.MISSING,
    default_factory=dataclasses.MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
) -> dataclasses.Field:
    """Declares an OTree field that is ignored by PyTree creation.

    This function has identical arguments and return as the
    :obj:`dataclasses.field` function.
    """

    metadata = {} if metadata is None else metadata
    metadata = {"ojax-field-type": FieldType.IGNORE, **metadata}
    kwargs = {
        "default": default,
        "default_factory": default_factory,
        "init": init,
        "repr": repr,
        "hash": hash,
        "compare": compare,
        "metadata": metadata,
    }
    return dataclasses.field(**kwargs)


OTree_T = TypeVar("OTree_T", bound="OTree")


class OTree(pureclass.PureClass):
    """Base "object-like" class for JAX which bundles data (as an immutable
    PyTree) and pure functions.

    The dataclass fields in this class are categorized into three field types:

    * auxiliary: metadata that defines the type of the PyTree. They can be
      non-numerical data and should stay static.
    * child: children of this PyTree that hold the numerical data. These are
      typically jax arrays and sub-PyTrees. JAX computations act on this part.
    * ignored: fields that are not part of the PyTree.

    .. warning::
      Ignored fields are not preserved with a flatten/unflatten transform,
      which is implicitly used by many JAX transforms and functions.

    You can explicitly declare the category of each field with
    :meth:`ojax.aux`, :meth:`ojax.child` and :meth:`ojax.ignore`. Otherwise, it
    is inferred based on the annotated type of the field: subclasses of
    :class:`jax.Array` and :class:`ojax.OTree` are assumed to be child
    fields and the rest are assumed to be auxiliary fields. Example usage:

    .. code-block:: python

        class MyConv(ojax.OTree):
            out_channels: int  # inferred to be aux
            kernel_size: int = ojax.aux()  # declared to be aux
            weight: jax.numpy.ndarray  # inferred to be child
            bias: jax.Array = ojax.child(default=None)  # declared child
    """

    def __init_subclass__(cls, **kwargs):
        """Make each subclass a :class:`ojax.PureClass` and a PyTree."""
        
        purecls = super().__init_subclass__(**kwargs)
        return jax.tree_util.register_pytree_node_class(purecls)

    def update(self: OTree_T, **kwargs) -> OTree_T:
        """Create a new version of this OTree instance with updated children.

        This method only updates the numerical data and will keep the OTree 
        structure and the metadata intact. It is the intended method to 
        update the content without changing the PyTree type. If you need to 
        create a new OTree with a different metadata / altered structure, use
        :py:meth:`ojax.new` instead.

        .. warning::
          ``None`` is a special empty PyTree container. Thus updating numerical
          fields to ``None`` or vice versa will change the PyTree structure and
          trigger an error.

        Args:
            **kwargs: Keyword arguments specifying new values to be updated for
                the corresponding child fields.

        Returns:
            New OTree instance with updated children.
        """

        aux_names = set(
            f.name
            for f in dataclasses.fields(self)
            if self.infer_field_type(f) is FieldType.AUX
        )
        aux_args = aux_names.intersection(kwargs.keys())
        if len(aux_args) != 0:
            raise ValueError(
                f'update of keys {aux_args} not allowed, use "ojax.new()" '
                f"instead to create instance of {self.__class__.__name__} with"
                f" different PyTree structure."
            )
        for k, v in kwargs.items():
            old_v = object.__getattribute__(self, k)
            if jax.tree.structure(v) != jax.tree.structure(old_v):
                raise ValueError(
                    f"PyTree structure mismatch at key {k}: "
                    f"expected\n{jax.tree.structure(old_v)}\n"
                    f"received\n{jax.tree.structure(v)}"
                )
        return pureclass.new(self, **kwargs)

    def tree_flatten(
        self,
    ) -> tuple[tuple, tuple[tuple[tuple[str, int], ...], tuple]]:
        """Define the flatten behavior of OTree as a PyTree."""

        tree_leaves = []
        num_arrays = []
        aux_values = []
        for f in dataclasses.fields(self):
            name, ftype = f.name, self.infer_field_type(f)
            if ftype is FieldType.AUX:
                aux_values.append(getattr(self, name))
                num_arrays.append((name, 0))
            elif ftype is FieldType.CHILD:
                entry = getattr(self, name)
                entry_leaves, entry_aux = jax.tree.flatten(entry)
                tree_leaves.extend(entry_leaves)
                aux_values.append(entry_aux)
                num_arrays.append((name, len(entry_leaves)))
            elif ftype is FieldType.IGNORE:
                pass
            else:
                raise NotImplementedError
        return tuple(tree_leaves), (tuple(num_arrays), tuple(aux_values))

    @classmethod
    def tree_unflatten(
        cls: type[OTree_T],
        aux_data: tuple[tuple[tuple[str, int], ...], tuple],
        children: Sequence,
    ) -> OTree_T:
        """Define the unflatten behavior of OTree as a PyTree."""
        
        num_arrays, aux_values = aux_data
        num_arrays = dict(num_arrays)
        aux_values_iter = iter(aux_values)
        tree_children = {}
        offset = 0
        for f in dataclasses.fields(cls):
            name, ftype = f.name, cls.infer_field_type(f)
            if ftype is FieldType.AUX:
                tree_children[name] = next(aux_values_iter)
            elif ftype is FieldType.CHILD:
                count = num_arrays[name]
                child_leaves = children[offset : offset + count]
                tree_child = jax.tree.unflatten(
                    next(aux_values_iter), child_leaves
                )
                tree_children[name] = tree_child
                offset += count
            elif ftype is FieldType.IGNORE:
                pass
            else:
                raise NotImplementedError
        # alternative to cls.__init__ since it might be custom
        otree = cls.__new__(cls)
        otree.assign_(**tree_children)
        return otree

    @classmethod
    def infer_field_type(cls, f: dataclasses.Field) -> FieldType:
        """Infer the OJAX field type from a :class:`dataclasses.Field` object.

        When :class:`ojax.FieldType` is unspecified (when
        :py:meth:`get_field_type` returns ``None``), the annotated ``f.type``
        is used for inference: for subclasses of :class:`jax.Array` and
        :class:`ojax.OTree` are assumed to be child fields and the rest are
        assumed to be auxiliary fields. You can override this method through
        subclass inheritance to change the inference logic.

        Args:
            f: a field of :class:`OTree`.

        Returns:
            The inferred :class:`ojax.FieldType`.
        """

        try:
            f_type = get_field_type(f)
            if f_type is not None:
                return f_type
            elif issubclass(f.type, (OTree, jax.Array)):
                return FieldType.CHILD
            else:
                return FieldType.AUX
        except TypeError:  # issubclass fails if f.type is not a proper class
            return FieldType.AUX
