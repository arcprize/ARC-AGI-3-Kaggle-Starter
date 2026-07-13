"""Run `agent/my_agent.py` locally against a real ARC-AGI-3 game.

This is the fast inner-loop: no Docker, no Kaggle round-trip. Uses the
`arc-agi` PyPI package to host the game engine and the ARC-AGI-3-Agents
framework's `Agent.main()` loop to drive it exactly like the Kaggle
gateway does, just in-process.

Usage:
    .venv/bin/python scripts/play_local.py --game ls20 --max-steps 200
    .venv/bin/python scripts/play_local.py --list
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():
    raise SystemExit(f"Framework not found at {VENDOR}. Run `make setup` first.")
sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode

_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def load_my_agent_class():
    """Import MyAgent from agent/my_agent.py via importlib."""
    spec = importlib.util.spec_from_file_location(
        "user_agent_module", ROOT / "agent" / "my_agent.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("Could not load agent/my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "MyAgent"):
        raise SystemExit("agent/my_agent.py must define a class named `MyAgent`")
    return module.MyAgent


def _configure_local_runner_env(llm_provider: str) -> None:
    os.environ["AGI_LLM_PROVIDER"] = (
        str(llm_provider or "local").strip().lower() or "local"
    )
    preserve_proxy = str(
        os.environ.get("AGI_PLAY_LOCAL_PRESERVE_PROXY", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if preserve_proxy:
        return
    for env_name in _PROXY_ENV_VARS:
        os.environ.pop(env_name, None)
    no_proxy_entries = [
        entry.strip()
        for entry in str(os.environ.get("NO_PROXY", "") or "").split(",")
        if entry.strip()
    ]
    for host in ("localhost", "127.0.0.1"):
        if host not in no_proxy_entries:
            no_proxy_entries.append(host)
    os.environ["NO_PROXY"] = ",".join(no_proxy_entries)
    os.environ["no_proxy"] = os.environ["NO_PROXY"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default=None,
                   help="Game id to play. If omitted, plays ALL available games "
                        "(mirrors what Kaggle does in competition rerun). "
                        "Comma-separated list also accepted, e.g. ls20,vc33.")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Per-game cap on actions (overrides MyAgent.MAX_ACTIONS).")
    p.add_argument("--list", action="store_true",
                   help="List available games and exit.")
    p.add_argument("--render", default=None, choices=[None, "terminal"],
                   help="Optional terminal rendering each step.")
    p.add_argument("--record-dir", default=None,
                   help="Optional directory for PNG/GIF visual replay artifacts.")
    p.add_argument("--record-format", default="both", choices=["png", "gif", "both"],
                   help="Visual replay output format when --record-dir is set.")
    p.add_argument("--record-every", type=int, default=1,
                   help="Capture every Nth action; defaults to every action.")
    p.add_argument("--record-scale", type=int, default=8,
                   help="Nearest-neighbor pixel scale for recorded images.")
    p.add_argument("--record-labels", action=argparse.BooleanOptionalAction, default=True,
                   help="Add step/action/state labels to recorded frames.")
    p.add_argument(
        "--llm-provider",
        choices=["local", "openai"],
        default="local",
        help="LLM provider for local runs. Defaults to local so shell env does not leak OpenAI into play-local.",
    )
    args = p.parse_args()
    if args.record_every < 1 or args.record_scale < 1:
        p.error("--record-every and --record-scale must be at least 1")

    log_level = logging.DEBUG if os.environ.get("AGI_DEBUG_DECISIONS", "0").strip().lower() in {"1", "true", "yes", "on"} else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")
    if log_level == logging.DEBUG:
        logging.getLogger("agi_core.orch_or").setLevel(logging.DEBUG)
    _configure_local_runner_env(args.llm_provider)
    print(f"Using llm provider: {args.llm_provider}")

    # NORMAL = local execution; game source is downloaded on first call into
    # ./environment_files/ and cached for subsequent runs.
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    all_envs = arc.get_environments()

    if args.list:
        print(f"{len(all_envs)} environments:")
        for e in all_envs:
            print(f"  {e.game_id}: {getattr(e, 'title', '?')}")
        return

    # Resolve which games to play. `arc.make()` accepts the short game id
    # (e.g. "ls20") even though the EnvironmentInfo.game_id includes the
    # version suffix ("ls20-9607627b"), so we normalize to short ids.
    if args.game:
        wanted = {g.strip().split("-")[0] for g in args.game.split(",")}
        game_ids = [e.game_id.split("-")[0] for e in all_envs
                    if e.game_id.split("-")[0] in wanted]
        missing = wanted - set(game_ids)
        if missing:
            raise SystemExit(f"Unknown game id(s): {sorted(missing)}. Run --list.")
    else:
        game_ids = [e.game_id.split("-")[0] for e in all_envs]
        print(f"No --game specified; playing all {len(game_ids)} games "
              f"(this is what Kaggle does in competition rerun).\n")

    MyAgentCls = load_my_agent_class()
    if hasattr(MyAgentCls, "MAX_ACTIONS"):
        MyAgentCls.MAX_ACTIONS = min(MyAgentCls.MAX_ACTIONS, args.max_steps)

    per_game = []
    for i, game_id in enumerate(game_ids, 1):
        print(f"=== [{i}/{len(game_ids)}] {game_id} ===")
        env = arc.make(game_id, render_mode=args.render)
        if env is None:
            print(f"  could not create env for {game_id!r}, skipping")
            continue

        recorder = None
        if args.record_dir:
            from scripts.visual_replay import VisualReplayRecorder
            game_record_dir = Path(args.record_dir) / game_id
            recorder = VisualReplayRecorder(
                game_record_dir,
                fmt=args.record_format,
                every=args.record_every,
                scale=args.record_scale,
                labels=args.record_labels,
            )

        BaseAgent = MyAgentCls
        if recorder is not None:
            class RecordingAgent(BaseAgent):
                def take_action(self, action):
                    frame = super().take_action(action)
                    if frame is not None:
                        recorder.record(
                            frame,
                            action=getattr(action, "name", str(action)),
                            step=self.action_counter + 1,
                        )
                    return frame
            AgentCls = RecordingAgent
        else:
            AgentCls = BaseAgent

        agent = AgentCls(
            card_id="local-dev",
            game_id=game_id,
            agent_name=f"MyAgent.local.{game_id}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["local-dev"],
        )
        agent.main()

        artifacts = recorder.close() if recorder is not None else None

        final = agent.frames[-1]
        per_game.append((game_id, final.state, final.levels_completed,
                         agent.action_counter))
        print(f"  -> state={final.state}, levels_completed={final.levels_completed}, "
              f"actions={agent.action_counter}")
        if artifacts:
            print(f"  -> replay frames={artifacts['frames_dir']}, gif={artifacts['gif']}, "
                  f"timeline={artifacts['timeline']}")

    sc = arc.get_scorecard()
    print("\n========= SUMMARY =========")
    for gid, state, levels, actions in per_game:
        print(f"  {gid:8} levels={levels:3}  actions={actions:5}  state={state}")
    score_val = sc.score if hasattr(sc, "score") else sc
    print(f"\nAggregate scorecard score: {score_val}")


if __name__ == "__main__":
    main()
