"""Customized frozen dataclass for immutable computation."""

from __future__ import annotations
from typing import TypeVar
from typing_extensions import dataclass_transform
import warnings
import copy
from dataclasses import dataclass, fields


class NoAnnotationWarning(UserWarning):
    pass


def _is_magic_name(s: str) -> bool:
    return s.startswith("__") and s.endswith("__")


# get non property, non magical and non callable class variables
def _get_class_vars(cls: type) -> list[str]:
    return [
        m
        for m, v in cls.__dict__.items()
        if not (
            callable(getattr(cls, m))
            or _is_magic_name(m)
            or isinstance(v, property)
        )
    ]


# warn user about non-annotated class variables ambiguous for dataclasses
def _warn_no_anno_class_attrs(cls: type) -> None:
    anno = cls.__annotations__
    no_annos = tuple(n for n in _get_class_vars(cls) if n not in anno)
    if not no_annos:
        return
    warnings.warn(
        "Non-annotated class attributes are ignored by dataclass "
        f"{cls.__name__}: {no_annos}. Consider adding annotations and "
        "declaring class variables explicitly with typing.ClassVar instead.",
        NoAnnotationWarning,
    )


@dataclass_transform(frozen_default=True)
@dataclass(frozen=True, init=False)
class PureClass:
    """ "Record-type" base class with immutable and annotated dataclass fields.

    Direct attribute assignment with "=" is disabled to encourage the immutable
    paradigm. Use the ``ojax.new`` function to create new instances with
    updated field values instead of in-place updates.

    The ``.assign_`` method is provided to initialize fields in custom
    ``__init__`` methods. It is the low-level impure "dark magic" that normally
    should not be used by the end user in any other context.
    """

    def __init_subclass__(cls, **kwargs):
        """Make each subclass a frozen dataclass."""

        # warn user about potential missing annotation error
        _warn_no_anno_class_attrs(cls)
        return dataclass(frozen=True, init=False, **kwargs)(cls)

    def assign_(self, **kwargs) -> None:
        """Low-level in-place setting of PureClass instance fields.

        This should only be used during custom instance creation and before the
        first usage of the created instance. It is typically used in custom
        __init__ methods to replace the disabled direct assignment with "=".

        .. warning::
          End users should avoid using this method directly in other cases as
          it can easily break the immutable paradigm and cause potential bugs
          with JAX.

        Args:
            **kwargs: Keyword arguments specifying new values to be updated for
                the corresponding fields.
        """

        field_names = set(f.name for f in fields(self))
        arg_names = set(kwargs.keys())
        if not arg_names.issubset(field_names):
            raise ValueError(
                f"Unrecognized fields: {arg_names.difference(field_names)}"
            )
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)


PureClass_T = TypeVar("PureClass_T", bound=PureClass)


def new(pure_obj: PureClass_T, **kwargs) -> PureClass_T:
    """Shallow copy-based alternative to ``dataclasses.replace()``.

    This function circumvents the instance creation with another ``__init__``
    call. It allows direct updates of ``init=False`` fields and avoids many
    "bad surprises" for custom ``__init__`` functions.

    Args:
        pure_obj: An instance of ``PureClass`` to be updated.
        **kwargs: Keyword arguments specifying new values to be updated for
            the corresponding fields.

    Returns:
        The updated instance.
    """

    new_obj = copy.copy(pure_obj)
    new_obj.assign_(**kwargs)
    return new_obj
