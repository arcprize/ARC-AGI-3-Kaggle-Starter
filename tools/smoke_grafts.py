"""Execute the generated notebook's graft cell against the real shipped source.

WHY
---
``verify_submission_notebook.py`` proves the cell *parses*. That is not the same as
proving it *works*: the level probe reads eleven attribute names off ``GameRun`` and
wraps a method whose signature it must match, and the priors graft rebinds a module
attribute that must actually be the one the agent calls. Every one of those is a name
that could be wrong, and every one would fail nine hours into a submission where the
only symptom is a missing file or an unchanged prompt.

Both dependencies are present in this repo, so none of it has to be assumed:

  * ``taaf.game`` imports once ``imageio`` is stubbed (it is only used for diagnostics
    rendering, which this test never reaches);
  * ``inference.agent.tool_agent`` imports with no stubbing at all.

The cell source is **read out of the generated notebook**, not copied here, so this
tests the artifact Sam uploads rather than a paraphrase of it.

WHAT IS AND IS NOT COVERED
--------------------------
Covered: the wrapper's signature and idempotence, all eleven ``GameRun`` field names,
the JSONL row's shape, the ``base_actions_per_level=None`` fallback branch, and that
the agent's system prompt gains exactly the addendum. Not covered: that the priors
help — that needs a scored run, and no local harness can produce one because the
solver needs the GPU model.

Run:
    .venv/Scripts/python.exe tools/smoke_grafts.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB = REPO / "1.33 scored in arc agi 3 competiotn in kaggle" / "arc3-duck-v14-recovery.ipynb"
BUNDLE = REPO / "scratch" / "archive2_extracted" / "src"

sys.path.insert(0, str(BUNDLE / "tufa-arc-agi-framework" / "src"))
sys.path.insert(0, str(BUNDLE / "ARC3-Inference"))
sys.path.insert(0, str(BUNDLE / "taaf-grafts"))

# taaf.__init__ eagerly imports taaf.diagnostics, which pulls in plotting/stats
# packages this box does not have. They are only used to *render* diagnostics, which
# this test never reaches, so stub whatever is missing rather than install a
# scientific stack to check eleven attribute names. Stubs are discovered by asking
# the import to fail, so the list stays honest instead of being guessed.
def _import_with_stubs(name: str, limit: int = 12):
    import importlib

    stubbed: list[str] = []
    for _ in range(limit):
        try:
            module = importlib.import_module(name)
            if stubbed:
                print(f"[env] stubbed to allow import: {', '.join(stubbed)}")
            return module
        except ModuleNotFoundError as exc:
            missing = exc.name
            if not missing or missing in sys.modules:
                raise
            # A failed submodule import can leave ``taaf.game`` cached while
            # ``taaf`` itself never finished __init__. Retrying would then return a
            # half-imported module and the real failure would resurface later, in
            # the code under test, looking like the graft's fault. Purge the tree.
            root = name.split(".")[0]
            for cached in [k for k in sys.modules if k == root or k.startswith(root + ".")]:
                del sys.modules[cached]
            parts = missing.split(".")
            for i in range(len(parts)):
                dotted = ".".join(parts[: i + 1])
                if dotted not in sys.modules:
                    mod = types.ModuleType(dotted)
                    # __path__ makes the stub a *package*, so a later
                    # ``import imageio.v3`` resolves instead of raising
                    # "imageio is not a package" from inside the graft cell.
                    mod.__path__ = []  # type: ignore[attr-defined]
                    sys.modules[dotted] = mod
                if i:
                    setattr(sys.modules[".".join(parts[:i])], parts[i], sys.modules[dotted])
            stubbed.append(missing)
    raise SystemExit(f"could not import {name} after {limit} stubs: {stubbed}")


def graft_cell_source() -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "INSTALL_FAMILY_PRIORS" in src:
            return src
    raise SystemExit("graft cell not found in the generated notebook")


def main() -> int:
    tg = _import_with_stubs("taaf.game")
    tool_agent = _import_with_stubs("inference.agent.tool_agent")
    agent_ext = _import_with_stubs("taaf_grafts.agent_ext")

    stock_prompt_len = len(tool_agent._build_system_prompt(tool_output_tokens=4096))
    stock_finish = tg.Game.finish_game

    tmp = REPO / "scratch" / "graft_smoke"
    tmp.mkdir(parents=True, exist_ok=True)
    probe = tmp / "level_probe.jsonl"
    probe.unlink(missing_ok=True)

    # Exactly the namespace the notebook cell has when it runs: cell 1 imported json
    # and defined WORKING_DIR; nothing else in the cell reaches outside itself.
    ns: dict = {"json": json, "WORKING_DIR": tmp, "Path": Path, "__name__": "__notebook__"}
    exec(compile(graft_cell_source(), "graft_cell", "exec"), ns)

    problems: list[str] = []
    status = ns["SESSION_GRAFTS"]
    for name, value in status.items():
        if str(value).startswith("FAILED"):
            problems.append(f"{name}: {value}")

    # --- Graft B: the prompt actually grew, by exactly the addendum -------------
    addendum = ns["FAMILY_PRIORS_ADDENDUM"]
    grown = tool_agent._build_system_prompt(tool_output_tokens=4096)
    delta = len(grown) - stock_prompt_len
    print(f"[B] system prompt {stock_prompt_len} -> {len(grown)} chars (+{delta})")
    if delta != len(addendum):
        problems.append(f"prompt grew by {delta}, addendum is {len(addendum)}")
    if not grown.endswith(addendum):
        problems.append("addendum is not the suffix of the built system prompt")
    # Installing twice must not double the text — Kaggle users re-run cells.
    ns["_install_family_priors"]()
    if len(tool_agent._build_system_prompt(tool_output_tokens=4096)) != len(grown):
        problems.append("re-installing the priors graft duplicated the addendum")
    print("[B] idempotent on re-install: ok")

    # --- Graft A: a real GameRun through the real wrapper -----------------------
    if tg.Game.finish_game is stock_finish:
        problems.append("Game.finish_game was not wrapped")

    def make_run(base):
        run = tg.GameRun(game_id="probe01", number_of_levels=4, base_actions_per_level=base)
        run.state = "playing"
        run.actions_per_level = [6, 2994, 0, 0]
        run.levels_completed = 1
        run.started_at_monotonic = None
        return run

    class FakeGame:
        """Only what finish_game touches: a game_run and the defensive _finish_game."""

        def __init__(self, base):
            self.game_run = make_run(base)

        def _finish_game(self):
            return None

    for base in ([17, 38, 31, 16], None):
        fake = FakeGame(base)
        taaf_game_finish = tg.Game.finish_game
        taaf_game_finish(fake, generated_tokens=1234, uncached_input_tokens=99)
        # Second call: finish_game early-returns on a scored run, so no second row.
        taaf_game_finish(fake, generated_tokens=1234)

    rows = [json.loads(line) for line in probe.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[A] rows written: {len(rows)} (expected 2 — one per game, not per call)")
    if len(rows) != 2:
        problems.append(f"expected 2 rows, got {len(rows)}")
    else:
        want = {
            "game_id",
            "state",
            "levels_completed",
            "number_of_levels",
            "final_score",
            "actions_per_level",
            "base_actions_per_level",
            "wallclock_s",
            "turns",
            "generated_tokens",
            "note",
        }
        missing = want - set(rows[0])
        if missing:
            problems.append(f"row missing keys: {sorted(missing)}")
        if rows[0]["actions_per_level"] != [6, 2994, 0, 0]:
            problems.append(f"actions_per_level not carried through: {rows[0]['actions_per_level']}")
        if rows[0]["base_actions_per_level"] != [17, 38, 31, 16]:
            problems.append(f"baselines not carried through: {rows[0]['base_actions_per_level']}")
        if rows[1]["base_actions_per_level"] is not None:
            problems.append("the missing-baseline fallback did not record None")
        if rows[0]["final_score"] is None:
            problems.append("final_score was not computed")
        print(f"[A] row 0: {json.dumps(rows[0], sort_keys=True)}")
        print(f"[A] row 1 baselines: {rows[1]['base_actions_per_level']} (None = framework fallback)")

    # --- Graft D: the rider rides, and only when the stock note speaks -----------
    # Non-empty case: 100 actions against a baseline of 20 is over budget, so stock
    # emits header + budget line + commit reminder.
    loud = dict(level_number=0, actions_this_level=100, baseline_this_level=20, net_zero_actions=None)
    quiet = dict(level_number=0, actions_this_level=0, baseline_this_level=20, net_zero_actions=None)
    rider = ns["EFFICIENCY_NOTE_RIDER"]
    note_loud = agent_ext.build_efficiency_note(**loud)
    note_quiet = agent_ext.build_efficiency_note(**quiet)
    print(f"[D] note when over budget: {len(note_loud)} chars, rider present={rider in note_loud}")
    print(f"[D] note when nothing to report: {len(note_quiet)} chars (stock stays silent)")
    if rider not in note_loud:
        problems.append("efficiency rider missing from a non-empty note")
    if not note_loud.startswith("EFFICIENCY BUDGET"):
        problems.append("rider clobbered the stock note instead of appending to it")
    if note_quiet != "":
        problems.append(f"rider broke the stock silent path: {note_quiet!r}")
    ns["_install_efficiency_rider"]()
    if agent_ext.build_efficiency_note(**loud) != note_loud:
        problems.append("re-installing the efficiency rider duplicated the text")

    # --- Graft C: R2 dies, R1/R3 live -------------------------------------------
    # This is the graft where a half-failure is worse than not doing it at all: a
    # live R2 with recovery armed is precisely the config that scored 0.82.
    rec = _import_with_stubs("taaf_grafts.recovery")
    buttons = ["ACTION1", "ACTION2", "ACTION6"]
    stock_plan = rec.build_probe_plan(buttons)
    rec.PROBE_MAX_ACTIONS = 0
    killed_plan = rec.build_probe_plan(buttons)
    print(f"[C] probe plan: {len(stock_plan)} actions at stock, {len(killed_plan)} at PROBE_MAX_ACTIONS=0")
    if not stock_plan:
        problems.append("stock probe plan was already empty — the kill test proves nothing")
    if killed_plan:
        problems.append(f"R2 still plans {len(killed_plan)} actions at PROBE_MAX_ACTIONS=0")
    # R1/R3 must not be collateral damage: both are reached without build_probe_plan.
    for free_mechanism in ("REFRESH_LOCKIN_ACTS", "HANDOFF_MAX_CHARS", "WIPED_KNOWLEDGE_KEYS"):
        if not hasattr(rec, free_mechanism):
            problems.append(f"recovery is missing {free_mechanism} — R1/R3 may not be the mechanisms assumed")

    print()
    if problems:
        for p in problems:
            print(f"FAIL  {p}")
        return 1
    print("OK  all four grafts install and behave against the real shipped source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
