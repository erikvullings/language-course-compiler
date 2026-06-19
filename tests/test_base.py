"""Behavior of the provider-agnostic message normalization and models."""

from __future__ import annotations

import pytest

from course_compiler.llm import Message, Role, to_messages


def test_string_prompt_becomes_single_user_message():
    assert to_messages("hello") == [Message(Role.USER, "hello")]


def test_single_message_is_wrapped_in_a_list():
    msg = Message(Role.SYSTEM, "be terse")
    assert to_messages(msg) == [msg]


def test_sequence_of_messages_is_preserved():
    msgs = [Message(Role.SYSTEM, "ctx"), Message(Role.USER, "hi")]
    assert to_messages(msgs) == msgs


def test_empty_sequence_is_rejected():
    with pytest.raises(ValueError):
        to_messages([])


def test_message_serializes_role_as_plain_string():
    assert Message(Role.USER, "hi").as_dict() == {"role": "user", "content": "hi"}
