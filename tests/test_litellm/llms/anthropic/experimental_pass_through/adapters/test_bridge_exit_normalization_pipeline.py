"""
Pipeline-level tests for the bridge-layer exit normalization
(``adapters/handler.py``).

Unit tests on the pure helpers cannot detect the ``**kwargs`` →
``extra_kwargs`` plumbing being severed by an upstream refactor — the flag
would silently read ``None`` and 11101 would regress in production. These
tests drive the full ``anthropic_messages_handler`` path with a mocked
``litellm.acompletion`` and assert on the kwargs actually issued:

* F1: a trailing visible-text assistant (client prefill) must reach
  ``acompletion`` followed by a synthetic user continuation turn.
* F2: a ``model_info={"tool_choice_string_only": True}`` deployment must
  receive ``tool_choice == "required"`` instead of the object form; an
  untagged deployment must keep the object form.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
)

from litellm.types.utils import ModelResponse
from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
    anthropic_messages_handler,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    _SYNTHETIC_CONTINUATION_PROMPT,
)


def _mock_completion_response(model="hosted_vllm/test-model"):
    return ModelResponse(
        id="test-id",
        model=model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


@pytest.mark.asyncio
async def test_pipeline_trailing_assistant_gets_continuation_user():
    """F1 full-path: [user, assistant("partial")] issued to acompletion as
    [user, assistant("partial"), user(_SYNTHETIC_CONTINUATION_PROMPT)]."""
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _mock_completion_response()

        await anthropic_messages_handler(
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "partial"},
            ],
            model="hosted_vllm/test-model",
            custom_llm_provider="hosted_vllm",
            _skip_mcp_handler=True,
            is_async=True,
        )

        mock_acompletion.assert_called_once()
        sent_messages = mock_acompletion.call_args.kwargs["messages"]
        assert sent_messages[-1] == {"role": "user", "content": _SYNTHETIC_CONTINUATION_PROMPT}
        assert sent_messages[-2]["role"] == "assistant"
        assert sent_messages[-2]["content"] == "partial"


@pytest.mark.asyncio
async def test_pipeline_thinking_only_tail_dropped():
    """F1 full-path: content=null trailing assistant is dropped before
    acompletion."""
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _mock_completion_response()

        await anthropic_messages_handler(
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None},
            ],
            model="hosted_vllm/test-model",
            custom_llm_provider="hosted_vllm",
            _skip_mcp_handler=True,
            is_async=True,
        )

        mock_acompletion.assert_called_once()
        sent_messages = mock_acompletion.call_args.kwargs["messages"]
        assert len(sent_messages) == 1
        assert sent_messages[0]["role"] == "user"


@pytest.mark.asyncio
async def test_pipeline_anthropic_provider_untouched():
    """F1 full-path: tolerant provider keeps the prefill tail verbatim.

    ``anthropic`` routes to the native /v1/messages handler (never the
    bridge) via ``anthropic_messages_handler``; here we drive the bridge
    handler directly with a tolerant ``custom_llm_provider`` to prove the
    exit normalization is a no-op for it."""
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        LiteLLMMessagesToCompletionTransformationHandler,
    )

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _mock_completion_response("claude-sonnet-4-5")

        await LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "prefill"},
            ],
            model="claude-sonnet-4-5",
            custom_llm_provider="anthropic",
            _is_async=True,
            _skip_mcp_handler=True,
        )

        mock_acompletion.assert_called_once()
        sent_messages = mock_acompletion.call_args.kwargs["messages"]
        assert sent_messages[-1]["role"] == "assistant"
        assert sent_messages[-1]["content"] == "prefill"


@pytest.mark.asyncio
async def test_pipeline_tagged_deployment_flattens_tool_choice():
    """F2 full-path: model_info.tool_choice_string_only=True plumbs through
    kwargs → extra_kwargs and flattens object-form tool_choice to
    "required"."""
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _mock_completion_response()

        await anthropic_messages_handler(
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            model="hosted_vllm/test-model",
            custom_llm_provider="hosted_vllm",
            tool_choice={"type": "tool", "name": "my_tool"},
            tools=[
                {
                    "name": "my_tool",
                    "description": "test tool",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            model_info={"tool_choice_string_only": True},
            _skip_mcp_handler=True,
            is_async=True,
        )

        mock_acompletion.assert_called_once()
        assert mock_acompletion.call_args.kwargs["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_pipeline_untagged_deployment_keeps_object_tool_choice():
    """F2 full-path: untagged deployment keeps the translated object form —
    the flatten must be opt-in per deployment."""
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _mock_completion_response()

        await anthropic_messages_handler(
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            model="hosted_vllm/test-model",
            custom_llm_provider="hosted_vllm",
            tool_choice={"type": "tool", "name": "my_tool"},
            tools=[
                {
                    "name": "my_tool",
                    "description": "test tool",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            model_info={"tool_choice_string_only": False},
            _skip_mcp_handler=True,
            is_async=True,
        )

        mock_acompletion.assert_called_once()
        sent_tool_choice = mock_acompletion.call_args.kwargs["tool_choice"]
        assert isinstance(sent_tool_choice, dict)
        assert sent_tool_choice.get("type") == "function"
