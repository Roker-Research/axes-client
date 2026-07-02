from axes.agent import Complete, Finish, Message
from axes.agent.protocol import (
    PlanStepRequest,
    RunToolRequest,
    plan_result_adapter,
    request_adapter,
)


def test_request_discriminates_plan_step() -> None:
    raw = '{"verb":"plan_step","messages":[{"role":"user","content":"hi"}]}'
    req = request_adapter.validate_json(raw)
    assert isinstance(req, PlanStepRequest)
    assert req.messages[0].content == "hi"


def test_request_discriminates_run_tool() -> None:
    raw = (
        '{"verb":"run_tool","tool_call":'
        '{"name":"get_prices","arguments":{"ticker":"ZS"}}}'
    )
    req = request_adapter.validate_json(raw)
    assert isinstance(req, RunToolRequest)
    assert req.tool_call.name == "get_prices"


def test_plan_result_roundtrips_by_action() -> None:
    for value in (
        Complete(messages=[]),
        Message(content="hello"),
        Finish(reason="done"),
    ):
        restored = plan_result_adapter.validate_json(value.model_dump_json())
        assert type(restored) is type(value)


def test_output_is_carried_on_plan_results() -> None:
    payload = {"point": 1.0}
    assert (
        Complete.model_validate_json(
            Complete(messages=[], state=payload).model_dump_json()
        ).state
        == payload
    )
    assert (
        Message.model_validate_json(
            Message(content="x", state=payload).model_dump_json()
        ).state
        == payload
    )
    assert (
        Finish.model_validate_json(
            Finish(content=payload).model_dump_json()
        ).content
        == payload
    )


def test_output_defaults_to_none() -> None:
    assert Complete(messages=[]).state is None
    assert Message(content="x").state is None
    assert Finish().content is None


def test_extra_fields_are_rejected() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        request_adapter.validate_json('{"verb":"plan_step","bogus":1}')
