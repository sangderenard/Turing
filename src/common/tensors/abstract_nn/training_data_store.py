"""Persistent compiler-training corpus for tokenized program transformations.

The store retains raw program forms, lossless token sequences, per-position
token events, compiler lineage, transformation pairs, queued densification
commands, and versioned model-weight records.  SQLite keeps the artifact
portable and inspectable; no compiler command is executed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

import ast

from .token_encoder import encode_identity_tokens


SCHEMA_VERSION = 2


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrainingView:
    id: int
    program_id: int
    form: str
    payload_sha256: str
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class CompilerCommandRequest:
    id: int
    program_id: int
    source_form: str
    target_form: str
    command_name: str
    arguments: Mapping[str, Any]
    status: str


class CompilerTrainingDatabase:
    """Versioned training corpus and compiler-permutation work queue."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "CompilerTrainingDatabase":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _migrate(self) -> None:
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS corpus_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS programs (
            id INTEGER PRIMARY KEY,
            corpus_key TEXT NOT NULL UNIQUE,
            source_language TEXT NOT NULL,
            entrypoint TEXT NOT NULL,
            source_text TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS views (
            id INTEGER PRIMARY KEY,
            program_id INTEGER NOT NULL REFERENCES programs(id),
            form TEXT NOT NULL,
            schema_name TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            lexicon_revision INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            tokens_json TEXT NOT NULL,
            token_ids_json TEXT NOT NULL,
            context_json TEXT NOT NULL,
            context_sha256 TEXT NOT NULL,
            generator TEXT NOT NULL,
            generator_version TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(program_id, form, payload_sha256)
        );
        CREATE TABLE IF NOT EXISTS token_events (
            view_id INTEGER NOT NULL REFERENCES views(id) ON DELETE CASCADE,
            position INTEGER NOT NULL,
            token TEXT NOT NULL,
            token_id TEXT,
            PRIMARY KEY(view_id, position)
        );
        CREATE TABLE IF NOT EXISTS transformations (
            id INTEGER PRIMARY KEY,
            program_id INTEGER NOT NULL REFERENCES programs(id),
            source_view_id INTEGER NOT NULL REFERENCES views(id),
            target_view_id INTEGER NOT NULL REFERENCES views(id),
            transform_name TEXT NOT NULL,
            weight_key TEXT NOT NULL,
            compiler_command_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_view_id, target_view_id, transform_name)
        );
        CREATE TABLE IF NOT EXISTS compiler_commands (
            id INTEGER PRIMARY KEY,
            program_id INTEGER NOT NULL REFERENCES programs(id),
            source_form TEXT NOT NULL,
            target_form TEXT NOT NULL,
            command_name TEXT NOT NULL,
            arguments_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            result_view_id INTEGER REFERENCES views(id),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT,
            UNIQUE(program_id, source_form, target_form, command_name, status)
        );
        CREATE TABLE IF NOT EXISTS weight_sets (
            weight_key TEXT PRIMARY KEY,
            source_form TEXT NOT NULL,
            target_form TEXT NOT NULL,
            architecture_json TEXT NOT NULL,
            revision INTEGER NOT NULL,
            checkpoint_uri TEXT,
            metrics_json TEXT NOT NULL,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """)
        current = self.connection.execute(
            "SELECT value FROM corpus_meta WHERE key='schema_version'"
        ).fetchone()
        if current is not None and int(current[0]) > SCHEMA_VERSION:
            raise ValueError(f"training database schema {current[0]} is newer than {SCHEMA_VERSION}")
        command_columns = {
            row[1] for row in self.connection.execute(
                "PRAGMA table_info(compiler_commands)"
            )
        }
        for column, declaration in (
            ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
            ("started_at", "TEXT"),
            ("last_error_json", "TEXT NOT NULL DEFAULT '{}'"),
        ):
            if column not in command_columns:
                self.connection.execute(
                    f"ALTER TABLE compiler_commands ADD COLUMN {column} {declaration}"
                )
        self.connection.execute(
            "INSERT OR REPLACE INTO corpus_meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.connection.commit()
    def put_program(
        self,
        source_text: str,
        entrypoint: str,
        *,
        source_language: str = "python",
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        source_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        corpus_key = _digest({
            "language": source_language,
            "entrypoint": entrypoint,
            "source_sha256": source_sha256,
        })
        self.connection.execute(
            """INSERT OR IGNORE INTO programs
               (corpus_key, source_language, entrypoint, source_text,
                source_sha256, metadata_json) VALUES (?, ?, ?, ?, ?, ?)""",
            (corpus_key, source_language, entrypoint, source_text,
             source_sha256, _canonical(dict(metadata or {}))),
        )
        self.connection.commit()
        return int(self.connection.execute(
            "SELECT id FROM programs WHERE corpus_key=?", (corpus_key,),
        ).fetchone()[0])

    def put_view(
        self,
        program_id: int,
        form: str,
        payload: Any,
        tokens: Sequence[str],
        *,
        token_ids: Sequence[int] = (),
        context: Mapping[str, Any] | None = None,
        schema_name: str = "turing-compiler-view",
        schema_version: int = 1,
        lexicon_revision: int = 0,
        generator: str = "unknown",
        generator_version: str = "working-tree",
    ) -> TrainingView:
        payload_json = _canonical(payload)
        payload_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        context_json = _canonical(dict(context or {}))
        context_sha256 = hashlib.sha256(context_json.encode("utf-8")).hexdigest()
        token_tuple = tuple(map(str, tokens))
        token_id_tuple = tuple(map(int, token_ids))
        if token_id_tuple and len(token_id_tuple) != len(token_tuple):
            raise ValueError("token_ids must align one-for-one with tokens")
        self.connection.execute(
            """INSERT OR IGNORE INTO views
               (program_id, form, schema_name, schema_version,
                lexicon_revision, payload_json, payload_sha256, tokens_json,
                token_ids_json, context_json, context_sha256, generator,
                generator_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (int(program_id), str(form), str(schema_name), int(schema_version),
             int(lexicon_revision), payload_json, payload_sha256,
             _canonical(token_tuple), _canonical(token_id_tuple), context_json,
             context_sha256, str(generator), str(generator_version)),
        )
        row = self.connection.execute(
            "SELECT id FROM views WHERE program_id=? AND form=? AND payload_sha256=?",
            (int(program_id), str(form), payload_sha256),
        ).fetchone()
        view_id = int(row[0])
        self.connection.executemany(
            "INSERT OR REPLACE INTO token_events(view_id, position, token, token_id) VALUES (?, ?, ?, ?)",
            (
                (view_id, position, token,
                 str(token_id_tuple[position]) if token_id_tuple else None)
                for position, token in enumerate(token_tuple)
            ),
        )
        self.connection.commit()
        return TrainingView(
            view_id, int(program_id), str(form), payload_sha256,
            token_tuple, token_id_tuple,
        )

    def forms(self, program_id: int) -> tuple[str, ...]:
        return tuple(row[0] for row in self.connection.execute(
            "SELECT DISTINCT form FROM views WHERE program_id=? ORDER BY form",
            (int(program_id),),
        ))

    def link_views(
        self,
        program_id: int,
        source_view_id: int,
        target_view_id: int,
        transform_name: str,
        *,
        weight_key: str,
        compiler_command: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        self.connection.execute(
            """INSERT OR IGNORE INTO transformations
               (program_id, source_view_id, target_view_id, transform_name,
                weight_key, compiler_command_json, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (int(program_id), int(source_view_id), int(target_view_id),
             str(transform_name), str(weight_key),
             _canonical(dict(compiler_command or {})),
             _canonical(dict(metadata or {}))),
        )
        self.connection.commit()
        return int(self.connection.execute(
            """SELECT id FROM transformations
               WHERE source_view_id=? AND target_view_id=? AND transform_name=?""",
            (int(source_view_id), int(target_view_id), str(transform_name)),
        ).fetchone()[0])

    def request_compiler_view(
        self,
        program_id: int,
        source_form: str,
        target_form: str,
        command_name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> CompilerCommandRequest:
        self.connection.execute(
            """INSERT OR IGNORE INTO compiler_commands
               (program_id, source_form, target_form, command_name, arguments_json)
               VALUES (?, ?, ?, ?, ?)""",
            (int(program_id), str(source_form), str(target_form),
             str(command_name), _canonical(dict(arguments or {}))),
        )
        self.connection.commit()
        row = self.connection.execute(
            """SELECT * FROM compiler_commands WHERE program_id=?
               AND source_form=? AND target_form=? AND command_name=?
               AND status='pending'""",
            (int(program_id), str(source_form), str(target_form), str(command_name)),
        ).fetchone()
        return CompilerCommandRequest(
            int(row["id"]), int(row["program_id"]), row["source_form"],
            row["target_form"], row["command_name"],
            json.loads(row["arguments_json"]), row["status"],
        )

    def pending_commands(self) -> tuple[CompilerCommandRequest, ...]:
        return tuple(
            CompilerCommandRequest(
                int(row["id"]), int(row["program_id"]), row["source_form"],
                row["target_form"], row["command_name"],
                json.loads(row["arguments_json"]), row["status"],
            )
            for row in self.connection.execute(
                "SELECT * FROM compiler_commands WHERE status='pending' ORDER BY id"
            )
        )

    def complete_command(self, command_id: int, result_view_id: int) -> None:
        self.connection.execute(
            """UPDATE compiler_commands SET status='complete', result_view_id=?,
               completed_at=CURRENT_TIMESTAMP WHERE id=?""",
            (int(result_view_id), int(command_id)),
        )
        self.connection.commit()

    def claim_command(self, command_id: int) -> CompilerCommandRequest:
        """Atomically move one pending command to running."""

        self.connection.execute("BEGIN IMMEDIATE")
        cursor = self.connection.execute(
            """UPDATE compiler_commands SET status='running',
               attempt_count=attempt_count+1, started_at=CURRENT_TIMESTAMP
               WHERE id=? AND status='pending'""",
            (int(command_id),),
        )
        if cursor.rowcount != 1:
            self.connection.rollback()
            raise ValueError(f"compiler command {command_id} is not pending")
        row = self.connection.execute(
            "SELECT * FROM compiler_commands WHERE id=?", (int(command_id),),
        ).fetchone()
        self.connection.commit()
        return CompilerCommandRequest(
            int(row["id"]), int(row["program_id"]), row["source_form"],
            row["target_form"], row["command_name"],
            json.loads(row["arguments_json"]), row["status"],
        )

    def fail_command(self, command_id: int, error: Mapping[str, Any]) -> None:
        self.connection.execute(
            """UPDATE compiler_commands SET status='failed',
               last_error_json=?, completed_at=CURRENT_TIMESTAMP WHERE id=?""",
            (_canonical(dict(error)), int(command_id)),
        )
        self.connection.commit()

    def retry_command(self, command_id: int) -> None:
        self.connection.execute(
            """UPDATE compiler_commands SET status='pending', started_at=NULL,
               completed_at=NULL WHERE id=? AND status='failed'""",
            (int(command_id),),
        )
        self.connection.commit()

    def program_record(self, program_id: int) -> Mapping[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM programs WHERE id=?", (int(program_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown training program {program_id}")
        return {
            "id": int(row["id"]), "source_language": row["source_language"],
            "entrypoint": row["entrypoint"], "source_text": row["source_text"],
            "source_sha256": row["source_sha256"],
            "metadata": json.loads(row["metadata_json"]),
        }

    def latest_view(self, program_id: int, form: str) -> TrainingView:
        row = self.connection.execute(
            """SELECT * FROM views WHERE program_id=? AND form=?
               ORDER BY id DESC LIMIT 1""",
            (int(program_id), str(form)),
        ).fetchone()
        if row is None:
            raise KeyError(f"program {program_id} has no {form!r} view")
        return TrainingView(
            int(row["id"]), int(row["program_id"]), row["form"],
            row["payload_sha256"], tuple(json.loads(row["tokens_json"])),
            tuple(map(int, json.loads(row["token_ids_json"]))),
        )

    def put_weight_set(
        self,
        weight_key: str,
        source_form: str,
        target_form: str,
        architecture: Mapping[str, Any],
        *,
        revision: int = 0,
        checkpoint_uri: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        status: str = "stub",
    ) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO weight_sets
               (weight_key, source_form, target_form, architecture_json,
                revision, checkpoint_uri, metrics_json, status, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (str(weight_key), str(source_form), str(target_form),
             _canonical(dict(architecture)), int(revision), checkpoint_uri,
             _canonical(dict(metrics or {})), str(status)),
        )
        self.connection.commit()


def _training_value(value: Any) -> Any:
    """Normalize compiler data without retaining process-local repr addresses."""

    if isinstance(value, ast.AST):
        return {"ast": ast.dump(value, include_attributes=True)}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, str):
        address_repr = re.fullmatch(
            r"<([A-Za-z_][A-Za-z_0-9.]*) object at 0x[0-9A-Fa-f]+>",
            value,
        )
        return address_repr.group(1) if address_repr is not None else value
    if value is None or isinstance(value, (bool, float, int)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _training_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        items = tuple(_training_value(item) for item in value)
        return tuple(sorted(items, key=_canonical)) if isinstance(
            value, (set, frozenset)
        ) else items
    value_type = type(value)
    return {"compiler_type": f"{value_type.__module__}.{value_type.__qualname__}"}


def put_reduced_graph_view(
    database: CompilerTrainingDatabase,
    program_id: int,
    graph: Any,
    *,
    form: str = "process_graph",
    lexicon_revision: int = 0,
    generator: str = "reduce_abstract_tensor_topology",
) -> TrainingView:
    """Persist a canonical graph and its exact dense-slot token vectors."""

    graph = getattr(graph, "G", graph)
    graph_metadata = getattr(graph, "graph", {})
    token_chains = {
        int(value_id): tuple(map(str, tokens))
        for value_id, tokens in (
            graph_metadata.get("ssa_identity_tokens") or {}
        ).items()
    }
    nodes = tuple({
        "value_id": int(value_id), "data": _training_value(dict(data)),
    } for value_id, data in graph.nodes(data=True))
    edges = tuple({
        "source": int(source), "target": int(target),
        "data": _training_value(dict(data)),
    } for source, target, data in graph.edges(data=True))
    retained_metadata = {
        key: _training_value(graph_metadata[key])
        for key in (
            "canonical_value_ids", "ssa_identity_tokens",
            "identity_table", "ingestion_identity_table",
            "function_name", "function_parameters", "function_outputs",
            "class_table", "map_ir",
        )
        if key in graph_metadata
    }
    tokens = tuple(
        token
        for value_id in sorted(token_chains)
        for token in (
            "slot:start", f"slot:{value_id}",
            *token_chains[value_id], "slot:end",
        )
    )
    return database.put_view(
        int(program_id), str(form),
        {"nodes": nodes, "edges": edges, "metadata": retained_metadata},
        tokens,
        token_ids=tuple(
            encode_identity_tokens({"token": token}) for token in tokens
        ),
        context={
            "node_count": len(nodes), "edge_count": len(edges),
            "dense_slots": tuple(sorted(token_chains)),
        },
        lexicon_revision=int(lexicon_revision), generator=str(generator),
    )


__all__ = [
    "CompilerCommandRequest", "CompilerTrainingDatabase", "TrainingView",
    "put_reduced_graph_view",
]
