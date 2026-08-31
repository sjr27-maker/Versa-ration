import ast
import re
from pathlib import Path

import probe.memory as memory_module


def test_memory_module_has_no_delete():
    """LearnerFactStore/ThinkingStyleStore must be append-only. See
    CLAUDE.md invariant 10. Same AST-based scan as every other store's
    equivalent check: walk the module's AST rather than raw text so
    docstring prose that talks *about* the constraint doesn't trip it,
    and flag actual violations (DELETE in string literals,
    delete/remove-prefixed function names).
    """
    source = Path(memory_module.__file__).read_text()
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
                "the memory layer must be append-only (see CLAUDE.md)"
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lname = node.name.lower()
            assert not (
                lname.startswith("delete") or lname.startswith("remove")
            ), (
                f"function {node.name!r} at line {node.lineno} looks like a "
                "removal method — the memory layer must be append-only "
                "(see CLAUDE.md)"
            )


def test_memory_migrations_have_no_delete():
    migrations_dir = Path(memory_module.__file__).resolve().parent / "migrations"
    for name in ("030_learner_facts_and_thinking_style.sql", "031_turn_diagnostics_memory_fields.sql"):
        source = (migrations_dir / name).read_text()
        code_only = "\n".join(line.split("--", 1)[0] for line in source.splitlines())
        code_only = re.sub(r"\bON\s+DELETE\s+\w+", "", code_only, flags=re.IGNORECASE)
        assert not re.search(r"\bDELETE\b", code_only, re.IGNORECASE), (
            f"{name} contains a DELETE statement — the memory layer's "
            "tables must be append-only (see CLAUDE.md)"
        )
