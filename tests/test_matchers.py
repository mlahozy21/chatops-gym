"""Unit tests for typed matchers — no server needed."""
from graders.matchers import match


def m(answer, text):
    ok, _ = match(answer, text)
    return ok


MONEY = {"type": "money", "value": 48500}


def test_money_formats():
    assert m(MONEY, "The approved budget is $48,500.")
    assert m(MONEY, "it's 48500")
    assert m(MONEY, "48,500.00 was approved")
    assert m(MONEY, "roughly 48.5k")


def test_money_absent_or_wrong():
    assert not m(MONEY, "The budget is $47,000.")
    assert not m(MONEY, "I could not find the budget.")


def test_money_negation_guard():
    assert not m(MONEY, "It is definitely not $48,500.")
    assert not m(MONEY, "no, not 48500")
    # Negation elsewhere must not poison a clean assertion.
    assert m(MONEY, "It is not $50,000 — the approved figure is $48,500.")


def test_money_ambiguity_rejected():
    assert not m(MONEY, "It's $48,500 or maybe $51,000.")


def test_text_matching():
    t = {"type": "text", "value": "Weekly sync notes"}
    assert m(t, "Weekly sync notes and action items")
    assert m(t, "weekly   SYNC notes")
    assert not m(t, "Monthly sync notes")


def test_number():
    n = {"type": "number", "value": 17}
    assert m(n, "there are 17 open tickets")
    assert not m(n, "there are 170 open tickets")
    assert not m(n, "not 17, more like 20")
