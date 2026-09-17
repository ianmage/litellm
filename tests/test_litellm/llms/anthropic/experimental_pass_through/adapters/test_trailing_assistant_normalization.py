"""
Regression tests for the bridge-layer exit normalization in the Anthropic
``/v1/messages`` → ``/chat/completions`` adapter (``adapters/handler.py``).

Background — what was broken (see ``docs/Venus400_FIND.md``):
* F1: a trailing assistant message with visible text (client prefill) was
  forwarded verbatim to OpenAI-compatible string-only gateways, which reject
  it with 400 error 11133/11153 ("conversation must end with a user message").
  The fix appends a synthetic user continuation turn after a visible-text
  assistant tail, and drops a content-empty assistant tail (thinking-only
  translation artifacts).
* F2: object-form ``tool_choice`` (``{"type": "function", ...}``) was
  forwarded verbatim to string-only gateways, which reject it with 400 error
  11101. The fix flattens it to ``"required"`` when the deployment is tagged
  ``model_info.tool_choice_string_only: true`` in config.yaml.
* F3: a trailing assistant with unfinished ``tool_calls`` (error 11148) is
  intentionally left untouched — no safe bridge-layer rewrite exists.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
)

from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    _SYNTHETIC_CONTINUATION_PROMPT,
    _deployment_tool_choice_string_only,
    _flatten_tool_choice_for_string_only_deployments as flatten,
    _normalize_trailing_assistant_for_prefill_intolerant_providers as normalize,
)


def test_text_tail_gets_continuation_appended():
    out = normalize([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}], "hosted_vllm")
    assert out[-1] == {"role": "user", "content": _SYNTHETIC_CONTINUATION_PROMPT}
    assert out[-2]["role"] == "assistant" and out[-2]["content"] == "x"


def test_previous_user_tail_no_consecutive_user():
    out = normalize([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}], "hosted_vllm")
    assert not (out[-2]["role"] == "user" and out[-1]["role"] == "user")


def test_thinking_only_tail_dropped():
    out = normalize([{"role": "user", "content": "hi"}, {"role": "assistant", "content": None}], "hosted_vllm")
    assert len(out) == 1 and out[0]["role"] == "user"


def test_empty_and_whitespace_tail_dropped():
    for c in ("", " "):
        out = normalize([{"role": "user", "content": "hi"}, {"role": "assistant", "content": c}], "hosted_vllm")
        assert len(out) == 1


def test_content_parts_list_tail_gets_continuation():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "partial"}, {"type": "text", "text": "  "}]},
    ]
    out = normalize(msgs, "hosted_vllm")
    assert out[-1]["role"] == "user" and out[-2]["content"] == msgs[1]["content"]


def test_all_whitespace_content_parts_tail_dropped():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": " "}, {"type": "text", "text": ""}]},
    ]
    out = normalize(msgs, "hosted_vllm")
    assert len(out) == 1


def test_tool_calls_tail_untouched():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": None, "tool_calls": [{"id": "t", "function": {"name": "f", "arguments": "{}"}}]}]
    assert normalize(msgs, "hosted_vllm") == msgs


def test_anthropic_provider_untouched():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "prefill"}]
    assert normalize(msgs, "anthropic") == msgs


def test_bedrock_vertex_tolerant():
    for p in ("bedrock", "vertex_ai"):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "p"}]
        assert normalize(msgs, p) == msgs


def test_unverified_provider_normalizes():
    for p in ("sagemaker", "snowflake", "converse", None):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "p"}]
        out = normalize(msgs, p)
        assert out[-1]["role"] == "user" and out[-2]["content"] == "p"


def test_trailing_user_untouched():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "d"}, {"role": "user", "content": "c"}]
    assert normalize(msgs, "hosted_vllm") == msgs


def test_trailing_tool_untouched():
    msgs = [{"role": "tool", "tool_call_id": "t", "content": "r"}]
    assert normalize(msgs, "hosted_vllm") == msgs


def test_empty_messages_untouched():
    assert normalize([], "hosted_vllm") == []


def test_tagged_deployment_flattens_object_form():
    assert flatten({"type": "function", "function": {"name": "f"}}, True) == "required"


def test_untagged_deployment_keeps_object_form():
    obj = {"type": "function", "function": {"name": "f"}}
    assert flatten(obj, False) == obj


def test_string_form_passthrough():
    assert flatten("auto", True) == "auto"


def test_model_info_flag_parsing():
    assert _deployment_tool_choice_string_only({"tool_choice_string_only": True}) is True
    assert _deployment_tool_choice_string_only({"tool_choice_string_only": False}) is False
    assert _deployment_tool_choice_string_only({}) is False
    assert _deployment_tool_choice_string_only(None) is False
