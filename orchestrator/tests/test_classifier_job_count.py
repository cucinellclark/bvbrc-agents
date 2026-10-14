"""The intent classifier says how many jobs a message asks for (D4 Phase 1).

``job_count: "many"`` puts the service agent in multi-job mode: every
``submit_gowe_job`` is recorded instead of submitted, and the user confirms the
set with one question.  Misreading many as one gives today's one-job-per-message
behaviour; one as many costs a question.  Non-JSON falls back to ``one``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from service_agent.classifier import CLASSIFIER_SYSTEM_PROMPT, _parse_classifier_response  # noqa: E402
from service_agent.models import Intent  # noqa: E402


def test_default_is_one():
    assert Intent().job_count == "one"
    assert Intent().services_named == []


def test_named_services_are_many():
    intent = _parse_classifier_response(
        '{"action": "plan", "job_count": "many", '
        '"services_named": ["Taxonomic Classification", "Metagenomic Binning"], "confidence": 0.9}'
    )
    assert intent.job_count == "many"
    assert intent.services_named == ["Taxonomic Classification", "Metagenomic Binning"]


def test_all_services_is_many_with_nothing_named():
    intent = _parse_classifier_response('{"action": "plan", "job_count": "many", "services_named": []}')
    assert intent.job_count == "many"
    assert intent.services_named == []


def test_one_service_is_one():
    intent = _parse_classifier_response('{"action": "plan", "job_count": "one", "services_named": ["Genome Assembly"]}')
    assert intent.job_count == "one"


def test_missing_or_invalid_job_count_is_one():
    assert _parse_classifier_response('{"action": "plan"}').job_count == "one"
    assert _parse_classifier_response('{"action": "plan", "job_count": "several"}').job_count == "one"
    assert _parse_classifier_response('{"action": "plan", "services_named": "CGA"}').services_named == []


def test_non_json_is_one():
    intent = _parse_classifier_response("I think the user wants two jobs")
    assert intent.action == "plan"
    assert intent.job_count == "one"


def test_prompt_teaches_the_fields():
    assert "job_count" in CLASSIFIER_SYSTEM_PROMPT
    assert "services_named" in CLASSIFIER_SYSTEM_PROMPT
    # The one miss in the live measurement: two steps of one service is one job.
    assert "ONE job" in CLASSIFIER_SYSTEM_PROMPT
