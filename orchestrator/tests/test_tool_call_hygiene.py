"""Malformed tool emissions must not reach the user or the circuit breaker.

Both failures below were observed on 2026-10-08 across five real submissions:

  3 of 5  submit_gowe_job arguments arrived as unparseable JSON. parse_tool_calls
          stored them as {_raw: ...}, the dispatcher then failed on the function
          signature with "missing 2 required positional arguments", and that
          counted against the two-failure circuit breaker.
  2 of 5  the model emitted a tool call as plain CONTENT. The loop took it as
          the final answer, so the user was shown raw <tool_call> XML -- on one
          of them the job had already submitted successfully.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared import agent_utils  # noqa: E402
from shared.agent_utils import (  # noqa: E402
    _recover_tool_arguments,
    contains_tool_call_markup,
    get_response_content,
    parse_tool_calls,
    strip_tool_call_markup,
)
from shared.tools import execute_tool, is_invalid_arguments_result  # noqa: E402


class _Call(SimpleNamespace):
    pass


def _response(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _tc(name, arguments, cid="c1"):
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=arguments))


# ---------------------------------------------------------------- recovery

@pytest.mark.parametrize("raw,expect", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),          # fenced
    ('{"a": 1}{"b": 2}', {"a": 1}),                 # two objects, take the first
    ('{"a": 1} trailing prose', {"a": 1}),
    ('{"a": 1,"b":', None),                          # truncated: unrecoverable
    ('[1, 2]', None),                                # a list is not arguments
    ('not json', None),
    ('', None),
    (None, None),
])
def test_argument_recovery(raw, expect):
    assert _recover_tool_arguments(raw) == expect


def test_unrecoverable_args_are_flagged_not_passed_through():
    calls = parse_tool_calls(_response(tool_calls=[_tc("submit_gowe_job", '{"inputs": broken')]), _Call)
    assert len(calls) == 1
    # The old behaviour was {"_raw": ...}, which reached the tool and produced a
    # signature error naming the wrong problem.
    assert "_unparseable_arguments" in calls[0].arguments
    assert "_raw" not in calls[0].arguments


@pytest.mark.asyncio
async def test_execute_tool_refuses_unparseable_without_calling():
    called = False

    async def tool(**_kw):
        nonlocal called
        called = True
        return {"ok": True}

    r = await execute_tool(
        "submit_gowe_job",
        {"_unparseable_arguments": '{"inputs": broken'},
        {"submit_gowe_job": tool},
        inject_config=False,
        inject_headers=False,
    )
    assert not called, "the tool must not run on unparseable arguments"
    assert is_invalid_arguments_result(r)
    # The message must name the actual problem and say nothing happened, or the
    # model cannot correct itself.
    assert "not valid JSON" in r["error"]
    assert "Nothing was submitted" in r["error"]


@pytest.mark.asyncio
async def test_a_normal_call_is_not_flagged():
    async def tool(**_kw):
        return {"ok": True}

    r = await execute_tool("probe_data", {"x": 1}, {"probe_data": tool},
                           inject_config=False, inject_headers=False)
    assert r == {"ok": True}
    assert not is_invalid_arguments_result(r)


# ------------------------------------------------------------ markup hygiene

QWEN_XML = (
    "<tool_call>\n<function=submit_gowe_job>\n<parameter=inputs>\n"
    '{"contrasts": "Cd30 vs Cd0"}\n</parameter>\n</function>\n</tool_call>'
)


def test_detects_and_strips_markup():
    assert contains_tool_call_markup(f"Done.\n{QWEN_XML}\nBye")
    assert strip_tool_call_markup(f"Done.\n{QWEN_XML}\nBye") == "Done.\n\nBye"


def test_strips_an_unterminated_block():
    # A truncated emission is still markup; the fragment must not survive.
    assert strip_tool_call_markup("answer\n<tool_call>\n<function=foo>") == "answer"


def test_get_response_content_strips_markup():
    # This is the exact shape that reached the user on the RNASeq submission.
    assert get_response_content(_response(content=QWEN_XML)) is None
    assert get_response_content(_response(content=f"Submitted.\n{QWEN_XML}")) == "Submitted."


def test_clean_content_is_untouched():
    assert get_response_content(_response(content="A normal answer.")) == "A normal answer."
    assert get_response_content(_response(content=None)) is None
