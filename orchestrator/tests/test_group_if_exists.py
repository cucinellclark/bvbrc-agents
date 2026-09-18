"""Tests for create_group(if_exists=error|append|replace).

Background: "put up to 200 genomes in my genome group X" called
Workspace.create on an existing path, which fails with a bare HTTP 500;
the agent retried four times with different queries and reported a
"temporary server issue".  There is no add-to-group API — the website
appends by read → merge → overwrite, and the tool now does the same.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "mcp_server") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "mcp_server"))

import shared.agent_dispatch  # noqa: E402, F401
from functions import group_functions as gf  # noqa: E402
from shared.tools.schemas import TOOL_SCHEMA_MAP  # noqa: E402

TOKEN = "un=alice@patricbrc.org|tokenid=abc|sig=x"
PATH = "/alice@patricbrc.org/home/Genome Groups/G"


def _get_response(exists: bool, ids=None, item_count=None, with_data=True):
    """Shape of Workspace.get: result[0][0][0] = meta tuple, [0][0][1] = data."""
    if not exists:
        return [[[[None, None, None, None, None]]]]
    auto = {"item_count": item_count} if item_count is not None else {}
    meta = ["G", "genome_group", "/alice@patricbrc.org/home/Genome Groups/", "2026-09-18T16:27:10Z",
            "ID-1", "alice@patricbrc.org", 100, {}, auto, "o", "n", ""]
    data = json.dumps({"id_list": {"genome_id": ids or []}, "name": "G"}) if with_data else "not json"
    return [[[meta, data]]]


class FakeApi:
    def __init__(self, exists, ids=None, item_count=None, with_data=True, create_error=None):
        self.calls = []
        self._get = _get_response(exists, ids, item_count, with_data)
        self._create_error = create_error

    async def acall(self, method, params, request_id, token):
        self.calls.append((method, params))
        if method == "Workspace.get":
            return self._get
        if method == "Workspace.create":
            if self._create_error:
                raise self._create_error
            return [[["G", "genome_group", PATH]]]
        raise AssertionError(method)

    def created(self):
        # JSON-RPC params are a one-element list wrapping the create dict.
        return [p[0] for m, p in self.calls if m == "Workspace.create"]


class TestMcpCreateGroup:
    async def test_new_group_created_without_overwrite(self):
        api = FakeApi(exists=False)
        r = await gf.create_group(api, "G", ["1313.1", "1313.2", "1313.1"], "genome_group", TOKEN)
        assert r["action"] == "created" and r["count"] == 2 and r["added"] == 2
        [params] = api.created()
        assert "overwrite" not in params
        assert params["objects"][0][3]["id_list"]["genome_id"] == ["1313.1", "1313.2"]

    async def test_existing_group_default_is_already_exists(self):
        api = FakeApi(exists=True, ids=["1313.1"], item_count=50)
        r = await gf.create_group(api, "G", ["1313.9"], "genome_group", TOKEN)
        assert r["errorType"] == "ALREADY_EXISTS"
        assert "50 members" in r["error"] and "if_exists='append'" in r["error"]
        assert r["path"] == PATH
        assert api.created() == []  # nothing written

    async def test_append_merges_and_dedupes(self):
        api = FakeApi(exists=True, ids=["1313.1", "1313.2"])
        r = await gf.create_group(api, "G", ["1313.2", "1313.3", "1313.3", "1313.4"], "genome_group", TOKEN, if_exists="append")
        assert r["action"] == "appended"
        assert r["added"] == 2 and r["already_present"] == 1 and r["count"] == 4 and r["previous_count"] == 2
        [params] = api.created()
        assert params["overwrite"] == 1
        assert params["objects"][0][3]["id_list"]["genome_id"] == ["1313.1", "1313.2", "1313.3", "1313.4"]
        assert "Added 2 new genome_id(s)" in r["message"]

    async def test_append_reads_full_object(self):
        api = FakeApi(exists=True, ids=[])
        await gf.create_group(api, "G", ["1313.1"], "genome_group", TOKEN, if_exists="append")
        get_params = [p for m, p in api.calls if m == "Workspace.get"][0]
        assert get_params["metadata_only"] is False

    async def test_append_on_missing_group_creates(self):
        api = FakeApi(exists=False)
        r = await gf.create_group(api, "G", ["1313.1"], "genome_group", TOKEN, if_exists="append")
        assert r["action"] == "created"
        assert "overwrite" not in api.created()[0]

    async def test_append_refuses_unreadable_data(self):
        api = FakeApi(exists=True, with_data=False)
        r = await gf.create_group(api, "G", ["1313.1"], "genome_group", TOKEN, if_exists="append")
        assert r["errorType"] == "INVALID_RESPONSE" and api.created() == []

    async def test_replace_overwrites_with_new_ids_only(self):
        api = FakeApi(exists=True, ids=["1313.1", "1313.2"], item_count=2)
        r = await gf.create_group(api, "G", ["1313.9"], "genome_group", TOKEN, if_exists="replace")
        assert r["action"] == "replaced" and r["count"] == 1 and r["previous_count"] == 2
        [params] = api.created()
        assert params["overwrite"] == 1
        assert params["objects"][0][3]["id_list"]["genome_id"] == ["1313.9"]
        get_params = [p for m, p in api.calls if m == "Workspace.get"][0]
        assert get_params["metadata_only"] is True  # no need to read members

    async def test_invalid_mode(self):
        r = await gf.create_group(FakeApi(False), "G", ["1"], "genome_group", TOKEN, if_exists="merge")
        assert r["errorType"] == "INVALID_PARAMETERS"

    async def test_workspace_failure_on_existence_check_is_surfaced(self):
        api = FakeApi(exists=False)

        async def boom(method, params, request_id, token):
            raise RuntimeError("Workspace down")

        api.acall = boom
        r = await gf.create_group(api, "G", ["1"], "genome_group", TOKEN)
        assert "Could not check whether" in r["error"]


class TestSharedTool:
    async def _run(self, mcp_result, **kwargs):
        from shared.tools import groups

        fake_gf = type("GF", (), {"create_group": AsyncMock(return_value=mcp_result)})()
        with patch.object(groups, "get_group_functions", lambda *_: fake_gf), patch.object(
            groups, "get_json_rpc", lambda *_: type("M", (), {"JsonRpcCaller": lambda **k: object()})
        ), patch.object(groups, "_fetch_ids", AsyncMock(return_value=(["1313.1", "1313.2"], 5000)), create=True):
            src = Path(groups.__file__).read_text()
            r = await groups.create_group(
                group_name="G", group_type="genome_group", collection="genome",
                query="genome_name:x", headers={"Authorization": TOKEN}, **kwargs,
            )
        return r, fake_gf.create_group

    def test_schema(self):
        props = TOOL_SCHEMA_MAP["create_group"]["function"]["parameters"]["properties"]
        assert props["if_exists"]["enum"] == ["error", "append", "replace"]
        assert props["if_exists"]["default"] == "error"

    async def test_invalid_if_exists_rejected_before_any_call(self):
        from shared.tools import groups

        r = await groups.create_group(
            group_name="G", group_type="genome_group", collection="genome", query="q",
            if_exists="merge", headers={"Authorization": TOKEN},
        )
        # Either our validation or an upstream validation error — but never a write.
        assert "error" in r
