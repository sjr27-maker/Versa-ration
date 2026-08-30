import ast
import re
from pathlib import Path

import probe.disambiguate as disambiguate_module


def test_disambiguate_module_has_no_delete():
    """DisambiguationStore must be append-only. See CLAUDE.md invariant
    9. Same AST-based scan as test_option_store.py/test_branch_store.py's
    equivalent checks: walk the module's AST rather than raw text so
    docstring prose that talks *about* the constraint doesn't trip it,
    and flag actual violations (DELETE in string literals,
    delete/remove-prefixed function names).
    """
    source = Path(disambiguate_module.__file__).read_text()
    tree = ast.parse(source)

    docstring_nodes: set[int] = set()
    for parent in ast.walk(tree):
        if isinstance(
            parent,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            body = getattr(parent, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_nodes.add(id(body[0].value))

    delete_kw = re.compile(r"\bDELETE\b", re.IGNORECASE)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_nodes
        ):
            assert not delete_kw.search(node.value), (
                f"string literal at line {node.lineno} contains DELETE — "
                "the disambiguation store must be append-only (see CLAUDE.md)"
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lname = node.name.lower()
            assert not (
                lname.startswith("delete") or lname.startswith("remove")
            ), (
                f"function {node.name!r} at line {node.lineno} looks like a "
                "removal method — the disambiguation store must be "
                "append-only (see CLAUDE.md)"
            )


def test_disambiguation_migration_has_no_delete():
    migration = (
        Path(disambiguate_module.__file__).resolve().parent
        / "migrations"
        / "029_disambiguation.sql"
    )
    source = migration.read_text()
    # Strip SQL comments (-- ...) before scanning, same reasoning as the
    # module-level check excluding docstrings: prose that talks *about*
    # DELETE must not trip this. "ON DELETE RESTRICT"/"ON DELETE CASCADE"
    # are FK constraint clauses, not DML DELETE statements, and every
    # table here legitimately has them — strip those too.
    code_only = "\n".join(
        line.split("--", 1)[0] for line in source.splitlines()
    )
    code_only = re.sub(r"\bON\s+DELETE\s+\w+", "", code_only, flags=re.IGNORECASE)
    assert not re.search(r"\bDELETE\b", code_only, re.IGNORECASE), (
        "029_disambiguation.sql contains a DELETE statement — the "
        "disambiguation tables must be append-only (see CLAUDE.md)"
    )
