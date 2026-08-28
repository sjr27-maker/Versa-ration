import ast
import re
from pathlib import Path

import probe.diagnostics as diagnostics_module


def test_diagnostics_module_has_no_delete():
    """TurnDiagnosticsStore must be append-only. See CLAUDE.md
    invariant 7. Same AST-based scan as test_hypothesis_store.py's
    equivalent check: walk the module's AST rather than raw text so
    docstring prose that talks *about* the constraint doesn't trip it,
    and flag actual violations (DELETE in string literals,
    delete/remove-prefixed function names).
    """
    source = Path(diagnostics_module.__file__).read_text()
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
                "turn_diagnostics must be append-only (see CLAUDE.md)"
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lname = node.name.lower()
            assert not (
                lname.startswith("delete") or lname.startswith("remove")
            ), (
                f"function {node.name!r} at line {node.lineno} looks like a "
                "removal method — turn_diagnostics must be append-only "
                "(see CLAUDE.md)"
            )
