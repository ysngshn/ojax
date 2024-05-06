# Change log

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
