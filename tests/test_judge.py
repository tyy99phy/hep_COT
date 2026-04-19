"""Tests for the judge + structured answer parser."""

from __future__ import annotations

from hep_cot.study.judge import judge_answer
from hep_cot.study.runner import parse_structured_answer


def test_parse_fenced_json():
    text = """Here is my answer:
```json
{"short_answer": "foo", "key_values": [{"quantity": "x", "value": "3.2%", "source": "§6.2"}]}
```
Done.
"""
    out = parse_structured_answer(text)
    assert out is not None
    assert out["short_answer"] == "foo"


def test_parse_bare_json():
    text = '{"a": 1, "b": [1, 2, 3]}'
    out = parse_structured_answer(text)
    assert out == {"a": 1, "b": [1, 2, 3]}


def test_parse_no_json_returns_none():
    assert parse_structured_answer("no json here at all") is None


def _make_q(answer):
    return {
        "question_id": "q1",
        "category": "syst",
        "answer_parsed": answer,
        "tool_calls": [{"name": "search_text"}, {"name": "get_figure"}],
    }


def test_judge_l1_pass_on_identical():
    ref = _make_q(
        {
            "key_values": [
                {"quantity": "JES", "value": "3.2%", "source": "§6.2"},
                {"quantity": "lumi", "value": "2.3%", "source": "§6.3"},
            ],
            "citations": ["§6.2", "Fig 8"],
        }
    )
    student = _make_q(
        {
            "key_values": [
                {"quantity": "JES", "value": "3.2%", "source": "§6.2"},
                {"quantity": "lumi", "value": "2.3%", "source": "§6.3"},
            ],
            "citations": ["§6.2", "Fig 8"],
        }
    )

    rpt = judge_answer(ref, student)
    assert rpt.l1_pass is True
    assert rpt.l1_numbers["hit_rate"] == 1.0
    assert rpt.l1_citations["hit_rate"] == 1.0
    assert rpt.l2_tool_edit_distance == 0


def test_judge_l1_fail_on_wrong_number():
    ref = _make_q(
        {
            "key_values": [
                {"quantity": "JES", "value": "3.2%", "source": "§6.2"},
            ],
            "citations": ["§6.2"],
        }
    )
    student = _make_q(
        {
            "key_values": [
                {"quantity": "JES", "value": "5.0%", "source": "§6.2"},
            ],
            "citations": ["§6.2"],
        }
    )

    rpt = judge_answer(ref, student)
    assert rpt.l1_pass is False
    assert rpt.l1_numbers["hit_rate"] == 0.0
    assert "3.2" in (rpt.diff_summary or "")


def test_judge_citation_normalisation():
    """Ensure §, Section, Sec. all normalise the same way."""
    ref = _make_q({"citations": ["§6.2"], "key_values": []})
    student = _make_q({"citations": ["Section 6.2"], "key_values": []})

    rpt = judge_answer(ref, student)
    assert rpt.l1_citations["hit_rate"] == 1.0
