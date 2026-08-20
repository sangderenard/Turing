"""Versioned compiler-token experience, separate from compilation state."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import hashlib
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .token_encoder import encode_identity_tokens

SCHEMA = "turing-compiler-token-lexicon-v4"
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\s]")


def lexical_tokens(text: str) -> tuple[str, ...]:
    """Stable source-token split; no model vocabulary or cache is involved."""

    return tuple(_TOKEN.findall(text))


def structural_context_tokens(context: Mapping[str, Any]) -> tuple[str, ...]:
    """Encode compiler structure as an ordered, labeled training sequence.

    Field labels are tokens too: ``node_kind=Name`` and
    ``target.ast=Name`` must not collapse into the same observation.  The
    values are retained rather than replaced by a digest; a digest is only an
    index for this lossless context sequence.
    """

    tokens: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key in sorted(value, key=lambda key: str(key)):
                key = str(raw_key)
                visit(value[raw_key], f"{path}.{key}" if path else key)
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        tokens.append(f"field:{path or 'value'}")
        tokens.extend(f"value:{token}" for token in lexical_tokens(str(value)))

    visit(dict(context), "")
    return tuple(tokens)


@dataclass(frozen=True)
class CompilerTokenLexicon:
    """An immutable snapshot of the compiler's observed structural vocabulary.

    Token IDs are append-only. Recalibration updates frequency and adjacency
    statistics plus presentation rank, but never renumbers an established
    token; a compilation can therefore name the exact lexicon revision it
    used.
    """

    revision: int = 0
    token_ids: Mapping[str, int] | None = None
    counts: Mapping[str, int] | None = None
    transitions: Mapping[str, int] | None = None
    pending_tokens: Mapping[str, int] | None = None
    context_statistics: Mapping[str, Mapping[str, Any]] | None = None
    translation_steps: tuple[Mapping[str, Any], ...] = ()

    def token_id(self, token: str) -> int:
        """Return the token's reversible identity, never a dispenser number."""

        return encode_identity_tokens({"token": str(token)})

    def mint_pending(self, tokens: Iterable[str]) -> "CompilerTokenLexicon":
        """Record unknown source tokens for explicit later recalibration."""

        known = dict(self.token_ids or {})
        pending = dict(self.pending_tokens or {})
        for token in tokens:
            token = str(token)
            if token not in known:
                pending.setdefault(token, self.token_id(token))
        return CompilerTokenLexicon(
            self.revision, known, dict(self.counts or {}),
            dict(self.transitions or {}), pending,
            dict(self.context_statistics or {}),
            tuple(self.translation_steps),
        )

    def observe(self, documents: Iterable[str]) -> "CompilerTokenLexicon":
        """Observe source text when no compiler structure is available."""
        ids = dict(self.token_ids or {})
        counts = Counter(self.counts or {})
        transitions = Counter(self.transitions or {})
        for document in documents:
            tokens = lexical_tokens(document)
            for token in tokens:
                ids.setdefault(token, self.token_id(token))
                counts[token] += 1
            for left, right in zip(tokens, tokens[1:]):
                transitions[f"{left}\u0000{right}"] += 1
        return CompilerTokenLexicon(
            self.revision + 1, ids, dict(counts), dict(transitions), {},
            dict(self.context_statistics or {}),
            self._identity_translation_step(ids),
        )

    def observe_contexts(
        self, contexts: Iterable[Mapping[str, Any]],
    ) -> "CompilerTokenLexicon":
        """Observe labeled compiler context streams for a future model."""

        ids = dict(self.token_ids or {})
        counts = Counter(self.counts or {})
        transitions = Counter(self.transitions or {})
        context_statistics = {
            context_hash: dict(statistics)
            for context_hash, statistics in (self.context_statistics or {}).items()
        }
        for context in contexts:
            tokens = structural_context_tokens(context)
            token_ids = tuple(self.token_id(token) for token in tokens)
            canonical_context = json.dumps(
                dict(context), sort_keys=True, separators=(",", ":"),
            )
            context_hash = hashlib.sha256(
                canonical_context.encode("utf-8"),
            ).hexdigest()
            local_transitions = Counter(
                f"{left}\u0000{right}" for left, right in zip(tokens, tokens[1:])
            )
            previous = dict(context_statistics.get(context_hash) or {})
            merged_transitions = Counter(previous.get("transitions") or {})
            merged_transitions.update(local_transitions)
            context_statistics[context_hash] = {
                "context": json.loads(canonical_context),
                "tokens": tokens,
                "token_ids": token_ids,
                "count": int(previous.get("count", 0)) + 1,
                "transitions": dict(merged_transitions),
            }
            for token in tokens:
                ids.setdefault(token, self.token_id(token))
                counts[token] += 1
            for left, right in zip(tokens, tokens[1:]):
                transitions[f"{left}\u0000{right}"] += 1
        return CompilerTokenLexicon(
            self.revision + 1, ids, dict(counts), dict(transitions), {},
            context_statistics,
            self._identity_translation_step(ids),
        )

    def _identity_translation_step(
        self, target_token_ids: Mapping[str, int],
    ) -> tuple[Mapping[str, Any], ...]:
        """Record an upgrade edge even when deterministic IDs need no rewrite."""

        previous_tokens = set(self.token_ids or {})
        return (*self.translation_steps, {
            "from_revision": self.revision,
            "to_revision": self.revision + 1,
            "token_id_encoding": "canonical-json-base257-v1",
            "token_id_remap": {},
            "added_tokens": tuple(sorted(set(target_token_ids) - previous_tokens)),
            "context_field_renames": {},
        })

    def upgrade_document(self, document: Mapping[str, Any]) -> dict[str, Any]:
        """Bring a lexicon-annotated document through every recorded upgrade.

        Each step is data, not an implicit convention.  A future incompatible
        tokenizer or structural field change therefore has to add its remap to
        the lexicon record before documents can be declared current.
        """

        upgraded = dict(document)
        current = int(upgraded.get("lexicon_revision", 0))
        target = self.revision
        while current < target:
            step = next(
                (candidate for candidate in self.translation_steps
                 if int(candidate["from_revision"]) == current),
                None,
            )
            if step is None:
                raise ValueError(
                    f"no recorded translation from lexicon revision {current}",
                )
            remap = {int(old): int(new) for old, new in (
                step.get("token_id_remap") or {}).items()
            }
            if "context_token_ids" in upgraded:
                upgraded["context_token_ids"] = tuple(
                    remap.get(int(token_id), int(token_id))
                    for token_id in upgraded["context_token_ids"]
                )
            context = dict(upgraded.get("context") or {})
            for old, new in (step.get("context_field_renames") or {}).items():
                if old in context:
                    context[str(new)] = context.pop(old)
            if context:
                upgraded["context"] = context
            current = int(step["to_revision"])
        upgraded["lexicon_revision"] = target
        upgraded["lexicon_schema"] = SCHEMA
        return upgraded

    def ranked_tokens(self) -> tuple[str, ...]:
        return tuple(sorted(self.token_ids or {}, key=lambda token: (-int((self.counts or {}).get(token, 0)), token)))

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA, "revision": self.revision,
            "token_ids": dict(self.token_ids or {}),
            "counts": dict(self.counts or {}),
            "transitions": dict(self.transitions or {}),
            "pending_tokens": dict(self.pending_tokens or {}),
            "context_statistics": dict(self.context_statistics or {}),
            "translation_steps": list(self.translation_steps),
            "ranked_tokens": self.ranked_tokens(),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "CompilerTokenLexicon":
        if data.get("schema") not in {
            "turing-compiler-token-lexicon-v1",
            "turing-compiler-token-lexicon-v2",
            "turing-compiler-token-lexicon-v3",
            SCHEMA,
        }:
            raise ValueError(f"unsupported token lexicon schema {data.get('schema')!r}")
        context_statistics = {
            str(context_hash): {
                **dict(statistics),
                "tokens": tuple(statistics.get("tokens") or ()),
                "token_ids": tuple(
                    int(token_id) for token_id in (statistics.get("token_ids") or ())
                ),
            }
            for context_hash, statistics in (
                data.get("context_statistics") or {}
            ).items()
        }
        translation_steps = tuple({
            **dict(step),
            "added_tokens": tuple(step.get("added_tokens") or ()),
        } for step in (data.get("translation_steps") or ()))
        return cls(
            int(data.get("revision", 0)),
            dict(data.get("token_ids") or {}),
            dict(data.get("counts") or {}),
            dict(data.get("transitions") or {}),
            dict(data.get("pending_tokens") or {}),
            context_statistics,
            translation_steps,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CompilerTokenLexicon":
        candidate = Path(path)
        return cls.from_dict(json.loads(candidate.read_text(encoding="utf-8"))) if candidate.exists() else cls()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
