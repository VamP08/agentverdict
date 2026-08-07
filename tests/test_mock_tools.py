"""Mock tool fixtures: parsing, deterministic matching, and the bundled fixture file.

The registry is the agent-under-test's whole world, so these tests pin the matching
rules exactly: a replay that answers differently on the second run is worthless.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from agentverdict.agents.mock_tools import MockToolRegistry, ToolOutcome, default_fixture_path

BUNDLED_FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "mock_tools.json"

FIXTURE_DATA: dict[str, Any] = {
    "tools": {
        "lookup_order": {
            "rules": [
                {"when": {"order_id": "NW-1001"}, "result": {"status": "delivered"}},
                {
                    # Never reached for NW-1001: the broader rule above wins first.
                    "when": {"order_id": "NW-1001", "include_history": "true"},
                    "result": {"status": "delivered", "history": ["packed", "shipped"]},
                },
                {"when": {"order_id": 2002}, "result": {"status": "in_transit"}},
            ],
            "default": {"result": {"error": "order_not_found"}, "is_error": True},
        },
        # No default: an unmatched call falls through to the generic error outcome.
        "reset_password": {"rules": [{"when": {"user": "amy"}, "result": {"reset": True}}]},
    }
}

MALFORMED_FIXTURES: list[tuple[Any, str]] = [
    ("not a mapping at all", "must be a JSON object"),
    ({"tools": "nope"}, "must be an object of tool name"),
    ({"tools": {"": {"rules": []}}}, "non-empty strings"),
    ({"tools": {"t": {"rules": {}}}}, '"rules" must be a list'),
    ({"tools": {"t": {"rules": ["not-an-object"]}}}, "must be an object"),
    ({"tools": {"t": {"rules": [{"when": {"a": "1"}}]}}}, 'missing "result"'),
    ({"tools": {"t": {"rules": [{"when": {"a": {"n": 1}}, "result": 1}]}}}, "must be scalars"),
    ({"tools": {"t": {"default": {"result": 1, "is_error": "yes"}}}}, "must be true or false"),
]

PROBES: list[tuple[str, dict[str, Any]]] = [
    ("lookup_order", {"order_id": "NW-1001"}),
    ("lookup_order", {"order_id": "NW-9999"}),
    ("refund_order", {"order_id": "NW-4004"}),
    ("send_email", {"to": "maya.p@example.com", "subject": "Your refund"}),
    ("send_email", {"to": "someone-new@example.com"}),
    ("cancel_order", {"order_id": "NW-1001"}),
]


@pytest.fixture
def registry() -> MockToolRegistry:
    return MockToolRegistry.from_mapping(FIXTURE_DATA)


# --- matching ----------------------------------------------------------------


def test_rule_matches_on_argument_subset(registry: MockToolRegistry) -> None:
    outcome = registry.call(
        "lookup_order", {"order_id": "NW-1001", "reason": "damaged", "locale": "en"}
    )
    assert outcome == ToolOutcome(result={"status": "delivered"}, is_error=False)


def test_first_matching_rule_wins(registry: MockToolRegistry) -> None:
    outcome = registry.call("lookup_order", {"order_id": "NW-1001", "include_history": "true"})
    assert outcome.result == {"status": "delivered"}
    assert "history" not in outcome.result


def test_arguments_are_compared_as_trimmed_strings(registry: MockToolRegistry) -> None:
    assert registry.call("lookup_order", {"order_id": 2002}).result == {"status": "in_transit"}
    assert registry.call("lookup_order", {"order_id": " 2002 "}).result == {"status": "in_transit"}
    # Case-sensitive: a lowercased id is a different id, so it falls through.
    assert registry.call("lookup_order", {"order_id": "nw-1001"}).is_error is True


def test_falls_through_to_the_default_outcome(registry: MockToolRegistry) -> None:
    outcome = registry.call("lookup_order", {"order_id": "NW-9999"})
    assert outcome.is_error is True
    assert outcome.result == {"error": "order_not_found"}


def test_missing_argument_key_never_matches(registry: MockToolRegistry) -> None:
    assert registry.call("lookup_order", {}).result == {"error": "order_not_found"}
    assert registry.call("lookup_order").result == {"error": "order_not_found"}


def test_tool_without_a_default_reports_an_unmatched_call(registry: MockToolRegistry) -> None:
    outcome = registry.call("reset_password", {"user": "bob"})
    assert outcome.is_error is True
    assert "no fixture matched" in outcome.result["error"]
    assert outcome.result["arguments"] == {"user": "bob"}


def test_unknown_tool_returns_an_error_outcome(registry: MockToolRegistry) -> None:
    outcome = registry.call("cancel_order", {"order_id": "NW-1001"})
    assert outcome.is_error is True
    assert outcome.result["error"] == "unknown tool: cancel_order"
    assert outcome.result["available_tools"] == ["lookup_order", "reset_password"]

    empty = MockToolRegistry.empty()
    assert len(empty) == 0
    assert empty.call("lookup_order", {}).result["available_tools"] == []


def test_repeated_calls_are_identical(registry: MockToolRegistry) -> None:
    """Replays are only reproducible if the tools never vary between calls."""
    for name, arguments in PROBES:
        outcomes = [registry.call(name, dict(arguments)) for _ in range(3)]
        assert outcomes[0] == outcomes[1] == outcomes[2]
        rendered = {json.dumps(o.result, sort_keys=True, default=str) for o in outcomes}
        assert len(rendered) == 1


# --- parsing -----------------------------------------------------------------


@pytest.mark.parametrize(("data", "match"), MALFORMED_FIXTURES)
def test_malformed_fixtures_raise_value_error(data: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MockToolRegistry.from_mapping(data)


def test_bare_mapping_and_wrapped_mapping_agree() -> None:
    wrapped = MockToolRegistry.from_mapping(FIXTURE_DATA)
    bare = MockToolRegistry.from_mapping(FIXTURE_DATA["tools"])
    assert bare.tool_names() == wrapped.tool_names()
    assert bare.call("lookup_order", {"order_id": "NW-1001"}) == wrapped.call(
        "lookup_order", {"order_id": "NW-1001"}
    )


# --- the bundled fixture file ------------------------------------------------


def test_from_file_round_trips_the_bundled_fixture() -> None:
    from_disk = MockToolRegistry.from_file(BUNDLED_FIXTURE)
    from_parsed = MockToolRegistry.from_mapping(
        json.loads(BUNDLED_FIXTURE.read_text(encoding="utf-8"))
    )

    assert from_disk.tool_names() == ["lookup_order", "refund_order", "send_email"]
    assert from_parsed.tool_names() == from_disk.tool_names()
    for name, arguments in PROBES:
        assert from_disk.call(name, arguments) == from_parsed.call(name, arguments)

    known = from_disk.call("lookup_order", {"order_id": "NW-1001"})
    assert known.is_error is False
    assert known.result["status"] == "delivered"

    unknown = from_disk.call("lookup_order", {"order_id": "NW-9999"})
    assert unknown.is_error is True
    assert unknown.result["error"] == "order_not_found"

    # send_email's default is a success, so emailing a new address still "works".
    assert from_disk.call("send_email", {"to": "someone-new@example.com"}).is_error is False


def test_default_registry_loads_the_bundled_fixture() -> None:
    assert default_fixture_path().samefile(BUNDLED_FIXTURE)
    default = MockToolRegistry.default()
    explicit = MockToolRegistry.from_file(BUNDLED_FIXTURE)
    assert default.tool_names() == explicit.tool_names()
    assert default.call("refund_order", {"order_id": "NW-1001"}) == explicit.call(
        "refund_order", {"order_id": "NW-1001"}
    )


def test_from_file_rejects_missing_and_broken_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MockToolRegistry.from_file(tmp_path / "absent.json")

    not_json = tmp_path / "not-json.json"
    not_json.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        MockToolRegistry.from_file(not_json)

    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps({"tools": {"t": {"rules": {}}}}), encoding="utf-8")
    with pytest.raises(ValueError, match='"rules" must be a list') as excinfo:
        MockToolRegistry.from_file(malformed)
    assert str(malformed) in str(excinfo.value)
