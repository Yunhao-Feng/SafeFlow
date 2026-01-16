import os
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from traj import TraceTrack
from utils import load_config
from agent.default import DefaultAgent

console = Console()


def main(args):
    safe_agent = DefaultAgent(config=args)
    user_prompt = "Write me a hello-world python script."
    trace_track = TraceTrack(root_dir=args.output_dir, run_name=args.run_name, item_id="0trails")
    safe_agent.run(user_prompt=user_prompt, trace_track=trace_track)


if __name__ == "__main__":
    # 1. Start 👏
    console.print(Panel.fit("🚀 Safeflow Unified Evaluation", style="bold blue"))
    args = load_config(files=["default.yaml", "agent.yaml"])
    console.print("✅ Configuration loaded \n", style="green")
    console.print(args)

    # 2.Run
    main(args=args)

