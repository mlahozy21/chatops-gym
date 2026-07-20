"""Template determinism and consistency — no server needed."""
import json

from tasks.templates.channel_setup import TEMPLATE_L1, TEMPLATE_L2
from tasks.templates.rollup_reply import TEMPLATE as ROLLUP_REPLY
from tasks.templates.thread_reply import TEMPLATE as THREAD_REPLY
from validation.mutations import mutants_for

ALL = [TEMPLATE_L1, TEMPLATE_L2, THREAD_REPLY, ROLLUP_REPLY]


def test_determinism():
    for tpl in ALL:
        assert json.dumps(tpl.generate(7)) == json.dumps(tpl.generate(7))


def test_seed_variation():
    for tpl in ALL:
        assert tpl.generate(1)["prompt"] != tpl.generate(2)["prompt"] or \
               tpl.generate(1)["grader"] != tpl.generate(2)["grader"]


def test_schema_essentials():
    for tpl in ALL:
        t = tpl.generate(3)
        for key in ("task_id", "template_id", "seed", "difficulty", "prompt",
                    "setup", "grader", "oracle_solution"):
            assert key in t, f"{tpl.template_id} missing {key}"
        assert t["oracle_solution"][-1]["tool"] == "finish"
        assert t["difficulty"] in ("L1", "L2", "L3", "L4")


def test_every_family_has_mutants():
    for tpl in ALL:
        muts = mutants_for(tpl.generate(4))
        assert len(muts) >= 3, f"{tpl.template_id}: need >=3 mutants, got {len(muts)}"
        names = [m.name for m in muts]
        assert len(names) == len(set(names))


def test_oracle_uses_only_registered_tools():
    from gym.tools import _build_tools
    registered = {t.name for t in _build_tools()}
    for tpl in ALL:
        t = tpl.generate(5)
        for step in t["oracle_solution"]:
            assert step["tool"] in registered, \
                f"oracle step uses unregistered tool {step['tool']}"


def test_distractors_never_equal_answer():
    for seed in range(1, 30):
        t = THREAD_REPLY.generate(seed)
        assert t["meta"]["value"] not in t["meta"]["distractors"]
