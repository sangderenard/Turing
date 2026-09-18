"""Explicitly update the compiler's versioned source-token experience."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.tensors.abstract_nn.token_lexicon import (
    CompilerTokenLexicon,
    lexical_tokens,
    structural_context_tokens,
)
from src.common.tensors.abstract_nn.training_data_store import (
    CompilerTrainingDatabase,
)


def source_contexts(source: str) -> Iterable[dict[str, Any]]:
    """Yield the structural experience available before graph lowering."""

    for node in ast.walk(ast.parse(source)):
        yield {
            "node_kind": type(node).__name__,
            "ast": ast.dump(node, include_attributes=False),
            "field_names": tuple(name for name, _value in ast.iter_fields(node)),
            "source_span": {
                "line": getattr(node, "lineno", None),
                "column": getattr(node, "col_offset", None),
                "end_line": getattr(node, "end_lineno", None),
                "end_column": getattr(node, "end_col_offset", None),
            },
        }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", type=Path)
    parser.add_argument("--lexicon", type=Path, default=ROOT / "build" / "compiler_token_lexicon.json")
    parser.add_argument(
        "--database", type=Path,
        default=ROOT / "build" / "compiler_training.sqlite3",
        help="persistent training corpus receiving source and AST token views",
    )
    parser.add_argument(
        "--upgrade-document", type=Path,
        help="upgrade a lexicon-annotated JSON document through recorded revisions",
    )
    parser.add_argument(
        "--out", type=Path,
        help="destination for --upgrade-document (required for safety)",
    )
    args = parser.parse_args()
    if args.upgrade_document is not None:
        if args.sources:
            parser.error("sources and --upgrade-document are mutually exclusive")
        if args.out is None:
            parser.error("--upgrade-document requires --out")
        if not args.upgrade_document.is_file():
            parser.error(f"missing document: {args.upgrade_document}")
        document = json.loads(args.upgrade_document.read_text(encoding="utf-8"))
        upgraded = CompilerTokenLexicon.load(args.lexicon).upgrade_document(document)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(upgraded, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"upgraded revision {document.get('lexicon_revision', 0)} -> {upgraded['lexicon_revision']}: {args.out}")
        return 0
    if not args.sources:
        parser.error("provide source files or --upgrade-document")
    missing = [str(path) for path in args.sources if not path.is_file()]
    if missing:
        parser.error("missing source files: " + ", ".join(missing))
    documents = tuple(path.read_text(encoding="utf-8") for path in args.sources)
    contexts = tuple(
        context
        for document in documents
        for context in source_contexts(document)
    )
    lexicon = (
        CompilerTokenLexicon.load(args.lexicon)
        .observe(documents)
        .observe_contexts(contexts)
    )
    lexicon.save(args.lexicon)
    with CompilerTrainingDatabase(args.database) as database:
        for path, document in zip(args.sources, documents):
            program_id = database.put_program(
                document, path.stem,
                metadata={"source_path": str(path)},
            )
            source_tokens = lexical_tokens(document)
            source_view = database.put_view(
                program_id, "source", {"text": document}, source_tokens,
                token_ids=tuple(lexicon.token_ids[token] for token in source_tokens),
                lexicon_revision=lexicon.revision,
                generator="recalibrate_compiler_token_lexicon",
            )
            document_contexts = tuple(source_contexts(document))
            ast_tokens = tuple(
                token
                for context in document_contexts
                for token in (
                    "context:start",
                    *structural_context_tokens(context),
                    "context:end",
                )
            )
            ast_view = database.put_view(
                program_id, "python_ast", document_contexts, ast_tokens,
                token_ids=tuple(lexicon.token_id(token) for token in ast_tokens),
                lexicon_revision=lexicon.revision,
                generator="python.ast",
                generator_version=sys.version.split()[0],
            )
            database.link_views(
                program_id, source_view.id, ast_view.id, "parse_python",
                weight_key=f"source->python_ast@{lexicon.revision}",
                compiler_command={"command": "parse_python", "entry": path.stem},
            )
    print(f"revision {lexicon.revision}: {len(lexicon.token_ids or {})} tokens -> {args.lexicon}")
    print(f"training corpus -> {args.database}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
