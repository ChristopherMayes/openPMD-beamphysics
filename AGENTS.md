# Agent instructions

## Repository

- The package source is `beamphysics/`. `pmd_beamphysics/` is a compatibility shim and should not be edited.
- Create the development environment with `conda env create -f environment.yml`, activate it with `conda activate beamphysics-dev`, then install the package with `pip install -e .`.
- Run the tests with `python -m pytest` from the repository root, in the activated `beamphysics-dev` environment.
- Run `pre-commit run --files <changed files>` before committing. Its hooks rewrite files: `ruff --fix`, `ruff format` and notebook output clearing. Do not run them on files with unrelated changes, and do not discard or reformat existing work.
- The documentation is built by mkdocs with mkdocstrings, using numpy-style docstrings. API pages are in `docs/api/`, and example notebooks are in `docs/examples/`.
- Notebooks are committed without outputs. `scripts/execute_notebooks.bash` executes them.

## Python

- Write numpy-style docstrings. A multi-line docstring starts on the line after the opening `"""`.
- Docstrings are rendered as Markdown. Use single backticks for code, not double backticks or Sphinx roles such as `:class:` and `:meth:`. Write examples as doctests (`>>>`), not `::` blocks. Write math with `$...$` and `$$...$$`.
- Give units in parentheses, such as `Longitudinal position (m)`, not in brackets.
- Keep docstrings concise: the purpose, any defining formula, the parameters, the return value and the exceptions raised. Derivations and measurements belong in PR descriptions or example notebooks.
- Comment only where the reason is not evident from the code, such as a hidden constraint, an invariant or a workaround. Do not restate what the code does.
- Use modern type annotations (Python 3.10 and later): `list[int]`, `dict[str, X]`, `X | None` and `X | Y`. Do not import `List`, `Dict`, `Optional` or `Union` from `typing`.

## Tests

- Prefer fixtures and `@pytest.mark.parametrize` to loops or repeated setup.
- Keep test modules flat, with module-level functions, unless a class groups shared state or setup.

## Git

- Keep commit messages terse.
