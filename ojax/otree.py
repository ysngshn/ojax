from __future__ import annotations
from typing import cast
from typing_extensions import Self
from collections.abc import Sequence
import abc
from dataclasses import field, Field, MISSING, fields as dc_fields
import jax
from . import pureclass


class OTreeField(Field, metaclass=abc.ABCMeta):
    """Abstract dataclasses.Field subclass for OTree fields."""

    pass


class Aux(OTreeField):
    """Field subclass for OTree auxiliary data that belong to ``PyTreeDef``."""

    pass


class Child(OTreeField):
    """Field subclass for a child PyTree node."""

    pass


class Ignore(OTreeField):
    """Field subclass that is ignored by PyTree creation"""

    pass


class Alien(OTreeField):
    """Field subclass that is incompatible with PyTree flatten"""

    pass


def fields(
    otree: OTree | type[OTree],
    field_type: type[OTreeField] | None = None,
    infer: bool = True,
) -> tuple[Field, ...]:
    """Convenience function extending ``dataclasses.fields`` that can filter
    fields by OTree field type.

    Args:
        otree: the OTree instance to examine.
        field_type: if not None, specifies the field type to filter the list of
            fields from the OTree.
        infer: determines if the field type should be inferred in case it is
            not available. Has no effect when ``field_type = None``.

    Returns:
        A tuple of fields from the given OTree.
    """

    if field_type is not None and not issubclass(field_type, OTreeField):
        raise ValueError(f"{field_type} is not a OTreeField subclass.")
    t_fields = dc_fields(otree)
    if field_type is None:
        return t_fields
    else:
        if infer:
            return tuple(
                f
                for f in t_fields
                if issubclass(otree.__infer_otree_field_type__(f), field_type)
            )
        else:
            return tuple(f for f in t_fields if isinstance(f, field_type))


def aux(
    *,
    default=MISSING,
    default_factory=MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=MISSING,
) -> Aux:
    """Declares an OTree field that holds auxiliary data as part of
    `PyTreeDef`.

    This function has identical arguments to :obj:`dataclasses.field` and
    returns a field of type :class:`ojax.Aux`.
    """
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("cannot specify both default and default_factory")
    return Aux(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
    )


def child(
    *,
    default=MISSING,
    default_factory=MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=MISSING,
) -> Child:
    """Declares an OTree field that holds a child PyTree node.

    This function has identical arguments to :obj:`dataclasses.field` and
    returns a field of type :class:`ojax.Child`.
    """
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("cannot specify both default and default_factory")
    return Child(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
    )


def ignore(
    *,
    default=MISSING,
    default_factory=MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=MISSING,
) -> Ignore:
    """Declares an OTree field that is ignored by PyTree creation.

    This function has identical arguments to :obj:`dataclasses.field` and
    returns a field of type :class:`ojax.Ignore`.
    """
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("cannot specify both default and default_factory")
    return Ignore(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
    )


def alien(
    *,
    default=MISSING,
    default_factory=MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=MISSING,
) -> Alien:
    """Declares an OTree field that is not part of PyTree and crashes the
    flatten operations if holds value that is not None.

    This function has identical arguments to :obj:`dataclasses.field` and
    returns a field of type :class:`ojax.Alien`.
    """
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("cannot specify both default and default_factory")
    return Alien(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
    )


class AlienException(Exception):
    """Raised when trying to flatten an Alien field that is not None."""

    pass


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
        # updating typing.dataclass_transform
        purecls.__dataclass_transform__["field_specifiers"] = (
            field,
            Field,
            aux,
            Aux,
            child,
            Child,
            Ignore,
            ignore,
            Alien,
            alien,
        )
        return jax.tree_util.register_pytree_node_class(purecls)

    def update(self: Self, **kwargs) -> Self:
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

        aux_args = set(
            f.name for f in fields(self, field_type=Aux)
        ).intersection(kwargs.keys())
        if len(aux_args) != 0:
            raise ValueError(
                f'update of keys {aux_args} not allowed, use "ojax.new()" '
                f"instead to create new instances of {self.__class__.__name__}"
                " with updated auxiliary fields."
            )
        child_names = set(f.name for f in fields(self, field_type=Child))
        for k, v in kwargs.items():
            if k not in child_names:
                continue
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
        for f in fields(self):
            name, ftype = f.name, self.__infer_otree_field_type__(f)
            if issubclass(ftype, Aux):
                aux_values.append(getattr(self, name))
                num_arrays.append((name, 0))
            elif issubclass(ftype, Child):
                entry = getattr(self, name)
                entry_leaves, entry_aux = jax.tree.flatten(entry)
                tree_leaves.extend(entry_leaves)
                aux_values.append(entry_aux)
                num_arrays.append((name, len(entry_leaves)))
            elif issubclass(ftype, Alien):
                if getattr(self, name) is not None:
                    raise AlienException(f"Cannot flatten alien field {name}.")
                else:
                    pass
            elif issubclass(ftype, Ignore):
                pass
            else:  # pragma: no cover
                raise NotImplementedError
        return tuple(tree_leaves), (tuple(num_arrays), tuple(aux_values))

    @classmethod
    def tree_unflatten(
        cls: type[Self],
        aux_data: tuple[tuple[tuple[str, int], ...], tuple],
        children: Sequence,
    ) -> Self:
        """Define the unflatten behavior of OTree as a PyTree."""

        num_arrays, aux_values = aux_data
        dict_num_arrays = dict(num_arrays)
        aux_values_iter = iter(aux_values)
        tree_children = {}
        offset = 0
        for f in fields(cls):
            name, ftype = f.name, cls.__infer_otree_field_type__(f)
            if issubclass(ftype, Aux):
                tree_children[name] = next(aux_values_iter)
            elif issubclass(ftype, Child):
                count = dict_num_arrays[name]
                child_leaves = children[offset : offset + count]
                tree_child = jax.tree.unflatten(
                    next(aux_values_iter),
                    child_leaves,
                )
                tree_children[name] = tree_child
                offset += count
            elif issubclass(ftype, Ignore):
                pass
            elif issubclass(ftype, Alien):
                tree_children[name] = None
            else:  # pragma: no cover
                raise NotImplementedError
        # alternative to cls.__init__ since it might be custom
        otree = cls.__new__(cls)
        otree.assign_(**tree_children)
        return otree

    @classmethod
    def __infer_otree_field_type__(cls, f: Field) -> type[OTreeField]:
        """Infer the OJAX field type from a :class:`dataclasses.Field` object.

        When `f` does not have specified OTree field type (not an instance of
        :class:`ojax.OTreeField`), the annotated ``f.type`` is used for
        inference: for subclasses of :class:`jax.Array` and :class:`ojax.OTree`
        are assumed to be child fields and the rest are assumed to be auxiliary
        fields. You can override this method through subclass inheritance to
        change the inference logic.

        Args:
            f: a field of :class:`OTree`.

        Returns:
            The inferred :class:`ojax.OTreeField`.
        """

        try:
            if isinstance(f, Aux):
                return Aux
            elif isinstance(f, Child):
                return Child
            elif isinstance(f, Ignore):
                return Ignore
            elif isinstance(f, Alien):
                return Alien
            elif issubclass(cast(type, f.type), (OTree, jax.Array)):
                return Child
            else:
                return Aux
        except TypeError:  # issubclass fails if f.type is not a proper class
            return Aux
