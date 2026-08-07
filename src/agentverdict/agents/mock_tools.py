"""Fixture-driven mock tools for the replay harness.

The tools an agent-under-test calls are *data*, not code: a new scenario needs a
new entry in ``examples/mock_tools.json``, never a new Python branch. Lookups are
purely deterministic — the same call always yields the same outcome — so a replay
can be re-run and compared step for step.

Fixture shape (see DESIGN.md)::

    {"tools": {"lookup_order": {
       "rules": [{"when": {"order_id": "NW-1001"},
                  "result": {"status": "delivered"}, "is_error": false}],
       "default": {"result": {"error": "Order not found"}, "is_error": true}}}}

The first rule whose ``when`` entries all equal the call's arguments wins
(compared as trimmed strings, case-sensitive); otherwise the tool's ``default``;
otherwise an error outcome naming the tool.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIXTURE_FILENAME = "mock_tools.json"
FIXTURE_RELATIVE_PATH = Path("examples") / FIXTURE_FILENAME

#: Copy shipped inside the distribution. The repo keeps the editable original at
#: ``examples/mock_tools.json``; the build copies it here so an installed wheel
#: (the Docker image, for instance) can still replay.
PACKAGED_FIXTURE_PATH = Path(__file__).resolve().parent / "data" / FIXTURE_FILENAME


@dataclass(frozen=True)
class ToolOutcome:
    """What a mock tool returned. ``result`` must be JSON-serializable."""

    result: Any
    is_error: bool = False


@dataclass(frozen=True)
class _Rule:
    """One fixture rule: a normalized argument match plus its outcome."""

    when: tuple[tuple[str, str], ...]
    outcome: ToolOutcome

    def matches(self, arguments: Mapping[str, Any]) -> bool:
        for key, expected in self.when:
            if key not in arguments:
                return False
            if _normalize(arguments[key]) != expected:
                return False
        return True


@dataclass(frozen=True)
class _ToolFixture:
    """All fixtures registered for a single tool name."""

    name: str
    rules: tuple[_Rule, ...] = ()
    default: ToolOutcome | None = None


class MockToolRegistry:
    """Answers tool calls from fixtures, with no side effects and no randomness."""

    def __init__(self, tools: Mapping[str, _ToolFixture] | None = None) -> None:
        self._tools: dict[str, _ToolFixture] = dict(tools or {})

    # --- construction --------------------------------------------------------

    @classmethod
    def empty(cls) -> MockToolRegistry:
        """A registry with no tools; every call reports an unknown tool."""
        return cls({})

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> MockToolRegistry:
        """Build a registry from already-parsed fixture data.

        Accepts either the documented ``{"tools": {...}}`` wrapper or a bare
        mapping of tool name -> fixture. Raises ``ValueError`` on malformed input.
        """
        if not isinstance(data, Mapping):
            raise ValueError("mock tool fixtures must be a JSON object")
        tools: dict[str, _ToolFixture] = {}
        for name, spec in _extract_tools(data).items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("tool names must be non-empty strings")
            tools[name] = _parse_tool(name, spec)
        return cls(tools)

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> MockToolRegistry:
        """Load fixtures from ``path``.

        ``None`` means the bundled ``examples/mock_tools.json``. If that bundled
        file is missing (a trimmed install), an empty registry is returned rather
        than raising, so a replay reports per-call errors instead of dying. An
        explicitly named file that does not exist is still an error.
        """
        if path is None:
            resolved = default_fixture_path()
            if not resolved.is_file():
                return cls.empty()
        else:
            resolved = Path(path)
            if not resolved.is_file():
                raise FileNotFoundError(f"mock tool fixture not found: {resolved}")
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{resolved}: fixture is not valid JSON: {exc}") from exc
        try:
            return cls.from_mapping(data)
        except ValueError as exc:
            raise ValueError(f"{resolved}: {exc}") from exc

    @classmethod
    def default(cls) -> MockToolRegistry:
        """The bundled fixture set (empty if the fixture file is unavailable)."""
        return cls.from_file(None)

    # --- lookup --------------------------------------------------------------

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> ToolOutcome:
        """Resolve one tool call. Never raises: failures come back as outcomes."""
        args: Mapping[str, Any] = arguments if isinstance(arguments, Mapping) else {}
        fixture = self._tools.get(name)
        if fixture is None:
            return ToolOutcome(
                result={
                    "error": f"unknown tool: {name}",
                    "available_tools": sorted(self._tools),
                },
                is_error=True,
            )
        for rule in fixture.rules:
            if rule.matches(args):
                return rule.outcome
        if fixture.default is not None:
            return fixture.default
        return ToolOutcome(
            result={
                "error": f"no fixture matched for tool {name}",
                "arguments": dict(args),
            },
            is_error=True,
        )

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"MockToolRegistry(tools={self.tool_names()!r})"


def default_fixture_path() -> Path:
    """Locate the mock tool fixtures, preferring the packaged copy.

    Looked for in order: the copy inside the installed package; ``examples/``
    above this module (a source checkout, where editing the original should take
    effect immediately); ``examples/`` under the working directory. When nothing
    matches, the packaged location is returned so callers can report a real path.
    """
    if PACKAGED_FIXTURE_PATH.is_file():
        return PACKAGED_FIXTURE_PATH
    for parent in Path(__file__).resolve().parents:
        candidate = parent / FIXTURE_RELATIVE_PATH
        if candidate.is_file():
            return candidate
    cwd_candidate = Path.cwd() / FIXTURE_RELATIVE_PATH
    if cwd_candidate.is_file():
        return cwd_candidate
    return PACKAGED_FIXTURE_PATH


def _normalize(value: Any) -> str:
    """Fixture comparison key: stringified and trimmed, case-sensitive."""
    return str(value).strip()


def _extract_tools(data: Mapping[str, Any]) -> Mapping[str, Any]:
    wrapped = data.get("tools")
    if isinstance(wrapped, Mapping):
        if not all(isinstance(spec, Mapping) for spec in wrapped.values()):
            raise ValueError('"tools" must map each tool name to a fixture object')
        return wrapped
    if "tools" in data:
        raise ValueError('"tools" must be an object of tool name -> fixture')
    if all(isinstance(spec, Mapping) for spec in data.values()):
        return data
    raise ValueError('fixtures must look like {"tools": {"<tool name>": {...}}}')


def _parse_tool(name: str, spec: Mapping[str, Any]) -> _ToolFixture:
    if not isinstance(spec, Mapping):
        raise ValueError(f"tool {name!r}: fixture must be an object")
    raw_rules = spec.get("rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError(f'tool {name!r}: "rules" must be a list')
    rules = tuple(_parse_rule(name, index, entry) for index, entry in enumerate(raw_rules))
    raw_default = spec.get("default")
    default = None
    if raw_default is not None:
        default = _parse_outcome(f"tool {name!r} default", raw_default)
    return _ToolFixture(name=name, rules=rules, default=default)


def _parse_rule(tool: str, index: int, entry: Any) -> _Rule:
    where = f"tool {tool!r} rules[{index}]"
    if not isinstance(entry, Mapping):
        raise ValueError(f"{where}: must be an object")
    raw_when = entry.get("when", {})
    if not isinstance(raw_when, Mapping):
        raise ValueError(f'{where}: "when" must be an object')
    when: list[tuple[str, str]] = []
    for key, value in raw_when.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f'{where}: "when" keys must be non-empty strings')
        if isinstance(value, Mapping | list):
            raise ValueError(f'{where}: "when" values must be scalars, got {type(value).__name__}')
        when.append((key, _normalize(value)))
    return _Rule(when=tuple(when), outcome=_parse_outcome(where, entry))


def _parse_outcome(where: str, entry: Any) -> ToolOutcome:
    if not isinstance(entry, Mapping):
        raise ValueError(f"{where}: must be an object")
    if "result" not in entry:
        raise ValueError(f'{where}: missing "result"')
    is_error = entry.get("is_error", False)
    if not isinstance(is_error, bool):
        raise ValueError(f'{where}: "is_error" must be true or false')
    return ToolOutcome(result=entry["result"], is_error=is_error)
