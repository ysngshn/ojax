# Change log

## ojax 3.1.0

Add new `Alien` field type to denote "incompatible" fields that should crash 
PyTree flatten / unflatten operations:

- New features:
  - Added the `ojax.Alien` Field class.
  - Added `ojax.alien` for alien field type declaration.
  - Added `ojax.AlienException` which will be raised when flattening / 
  unflattening PyTrees with alien fields.
- Other changes:
  - fixed wrong `NoAnnotationWarning` warnings when a `property` is defined in 
  `ojax.OTree`. 
  - Updated the documentation to include the new alien field type.
  - Added test coverage analysis and improved unit tests 

## ojax 3.0.0

Rework `ojax.OTree` field type representation. 

Using custom subclass of 
`dataclasses.Field` instead of setting custom metadata. This simplifies field 
type checking and allows for further subclassing. Specifically:

- Classes `Aux`, `Child`, `Ignore` are added which indicates the type of 
`OTree` field. They all inherit from the abstract base class `OTreeField` 
which is itself a subclass of `dataclasses.Field`.
- The function `ojax.get_field_type` and the enum class `ojax.FieldType` is 
removed since they are now unnecessary.

Note that users should still use functions `aux()`, `child()`, `ignore()` to 
declare field types since directly using classes could generate confusion for 
type checking. This is also in line with the standard `dataclasses` usage.

## ojax 2.0.2

Fixes unnecessary missing annotation warning for non-user-defined attributes.

## ojax 2.0.1

Change doc theme. Improve documentation.

- Other changes
  - Now uses the sphinx_book_theme for Sphinx documentation.
  - Finetune documentation.

## ojax 2.0.0

Rework API. Add unit tests.

- Breaking changes
  - Renamed `ojax.OTree.infer_field_type()` to 
  `ojax.OTree.__infer_otree_field_type__()`.
- New features
  - Added the `ojax.fields()` function to get the fields from an OTree. It can
  also return fields of a certain field type.
- Other changes
  - Renamed `test/` folder to `tests/`.
  - Added unit tests.
  - Added CHANGELOG.md (this file).

## ojax 1.0.0

First version.
