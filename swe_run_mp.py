#!/usr/bin/env python3
"""
Parallel SWE-bench task runner that handles repository preparation and agent execution.

Workflow per task:
1. Clone repository and checkout to base_commit
2. Apply test_patch to expose the bug
3. Prepare task description for agent (without leaking patch)
4. Run agent to fix the issue
5. Evaluate final results

Parallelization:
- Uses ProcessPoolExecutor to run multiple tasks in parallel (one process per task).
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import load_dataset
from rich.console import Console

from agent.default import DefaultAgent
from utils import load_config


class SWERunner:
    """Handles SWE-bench task execution workflow."""

    def __init__(self, output_dir: str = "./swe_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def prepare_repository(self, task_data: Dict[str, Any], work_dir: Path, console: Console) -> bool:
        repo_url = f"https://github.com/{task_data['repo']}.git"
        repo_name = task_data['repo'].split('/')[-1]
        repo_path = work_dir / repo_name

        try:
            # 1. Clone repository
            try:
                ssh_url = f"git@github.com:{task_data['repo']}.git"
                console.print(f"Cloning {ssh_url}...", style="cyan")
                subprocess.run(
                    ["git", "clone", ssh_url, str(repo_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except Exception:
                console.print(f"Cloning {repo_url}...", style="cyan")
                subprocess.run(
                    ["git", "clone", repo_url, str(repo_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            # 2. Checkout to base_commit
            console.print(f"Checking out base commit {task_data['base_commit']}...", style="cyan")
            subprocess.run(
                ["git", "checkout", task_data["base_commit"]],
                cwd=str(repo_path),
                check=True,
                capture_output=True,
                text=True,
            )

            # 3. Apply test_patch
            if task_data.get("test_patch"):
                console.print("Applying test patch...", style="cyan")
                test_patch_file = work_dir / "test_patch.diff"
                test_patch_file.write_text(task_data["test_patch"], encoding="utf-8")

                result = subprocess.run(
                    ["git", "apply", str(test_patch_file)],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    console.print(
                        f"[yellow]Warning[/yellow]: Test patch application failed: {result.stderr}"
                    )
                    # fallback to patch
                    result = subprocess.run(
                        ["patch", "-p1", "-i", str(test_patch_file)],
                        cwd=str(repo_path),
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        console.print("[red]Error[/red]: Both git apply and patch failed")
                        return False

            return True

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error preparing repository[/red]: {e}", style="red")
            return False

    def run_tests(self, repo_path: Path, test_commands: List[str]) -> Tuple[bool, str]:
        all_output = []
        all_passed = True

        for cmd in test_commands:
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", cmd, "-v"],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                output = (
                    f"Command: {cmd}\n"
                    f"Return code: {result.returncode}\n"
                    f"STDOUT:\n{result.stdout}\n"
                    f"STDERR:\n{result.stderr}\n"
                    + "-" * 80
                    + "\n"
                )
                all_output.append(output)
                if result.returncode != 0:
                    all_passed = False

            except subprocess.TimeoutExpired:
                all_output.append(f"Command: {cmd}\nTIMEOUT after 120 seconds\n")
                all_passed = False

        return all_passed, "\n".join(all_output)

    def evaluate_solution(self, repo_path: Path, task_data: Dict[str, Any], console: Console) -> Dict[str, Any]:
        results = {
            "fail_to_pass": {"tests": [], "success": False, "output": ""},
            "pass_to_pass": {"tests": [], "success": False, "output": ""},
        }

        try:
            fail_to_pass = json.loads(task_data.get("FAIL_TO_PASS", "[]"))
            pass_to_pass = json.loads(task_data.get("PASS_TO_PASS", "[]"))
        except json.JSONDecodeError:
            console.print("[red]Error parsing test lists[/red]")
            return results

        if fail_to_pass:
            console.print(f"Running {len(fail_to_pass)} FAIL_TO_PASS tests...", style="cyan")
            success, output = self.run_tests(repo_path, fail_to_pass)
            results["fail_to_pass"] = {"tests": fail_to_pass, "success": success, "output": output}

        if pass_to_pass:
            console.print(f"Running {len(pass_to_pass)} PASS_TO_PASS tests...", style="cyan")
            success, output = self.run_tests(repo_path, pass_to_pass)
            results["pass_to_pass"] = {"tests": pass_to_pass, "success": success, "output": output}

        return results

    def prepare_agent_task(self, task_data: Dict[str, Any], repo_path: Path) -> str:
        safe_task_data = {
            "repo": task_data["repo"],
            "instance_id": task_data["instance_id"],
            "problem_statement": task_data["problem_statement"],
            "repository_path": str(repo_path),
            "fail_to_pass_tests": json.loads(task_data.get("FAIL_TO_PASS", "[]")),
            "pass_to_pass_tests": json.loads(task_data.get("PASS_TO_PASS", "[]")),
            "hints": task_data.get("hints_text", ""),
            "difficulty": task_data.get("difficulty", "unknown"),
        }

        prompt = f"""# SWE Task: {task_data['instance_id']}

## Repository: {task_data['repo']}

## Problem Statement:
{task_data['problem_statement']}

## Repository Location:
The repository has been prepared at: {repo_path}

## Your Task:
1. Investigate the issue described in the problem statement
2. Identify the root cause of the problem
3. Implement a fix that resolves the issue
4. Test your solution to ensure it works

## Testing:
- You can run tests using pytest
- There are specific failing tests that should pass after your fix: {safe_task_data['fail_to_pass_tests']}
- There are existing passing tests that must continue to pass: {safe_task_data['pass_to_pass_tests']}

## Guidelines:
- Make minimal changes to fix the issue
- Do not modify test files unless absolutely necessary
- Ensure your changes don't break existing functionality
- Focus on the specific issue described in the problem statement

## Additional Information:
- Difficulty: {safe_task_data['difficulty']}
- Hints: {safe_task_data['hints'] or 'No specific hints provided'}
"""
        return prompt

    def run_swe_task(self, task_data: Dict[str, Any], config: Dict[str, Any], console: Console) -> Dict[str, Any]:
        instance_id = task_data["instance_id"]
        console.print(f"\n{'='*60}", style="bold blue")
        console.print(f"Running SWE Task: {instance_id}", style="bold cyan")
        console.print(f"Repository: {task_data['repo']}", style="cyan")
        console.print(f"{'='*60}", style="bold blue")

        task_dir = self.output_dir / instance_id
        task_dir.mkdir(exist_ok=True, parents=True)

        try:
            if not self.prepare_repository(task_data, task_dir, console):
                return {"success": False, "error": "Failed to prepare repository", "instance_id": instance_id}

            repo_name = task_data["repo"].split("/")[-1]
            repo_path = task_dir / repo_name

            console.print("Verifying that FAIL_TO_PASS tests are currently failing...", style="cyan")
            fail_to_pass = json.loads(task_data.get("FAIL_TO_PASS", "[]"))
            if fail_to_pass:
                initial_success, _ = self.run_tests(repo_path, fail_to_pass)
                if initial_success:
                    console.print(
                        "[yellow]WARNING[/yellow]: FAIL_TO_PASS tests are already passing - bug may not be reproduced"
                    )

            agent_prompt = self.prepare_agent_task(task_data, repo_path)

            console.print("Starting agent to fix the issue...", style="green")
            agent = DefaultAgent(config=config, item_id=instance_id, work_root=str(repo_path))

            try:
                agent.run(agent_prompt)
                agent_success = True
                agent_error = None
            except Exception as e:
                agent_success = False
                agent_error = str(e)
                console.print(f"[red]Agent execution failed[/red]: {e}")

            console.print("Evaluating final solution...", style="cyan")
            evaluation = self.evaluate_solution(repo_path, task_data, console)

            overall_success = (
                agent_success and evaluation["fail_to_pass"]["success"] and evaluation["pass_to_pass"]["success"]
            )

            results = {
                "success": overall_success,
                "instance_id": instance_id,
                "repo": task_data["repo"],
                "agent_success": agent_success,
                "agent_error": agent_error,
                "evaluation": evaluation,
                "repository_path": str(repo_path),
            }

            results_file = task_dir / "results.json"
            results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

            console.print(
                f"\nTask completed. Overall success: {overall_success}",
                style="bold green" if overall_success else "bold red",
            )
            console.print(f"Results saved to: {results_file}", style="blue")

            return results

        except Exception as e:
            console.print(f"[red]Error running SWE task[/red]: {e}")
            return {"success": False, "error": str(e), "instance_id": instance_id}


def run_one_task(task_row: Dict[str, Any], config_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Top-level function for multiprocessing (must be picklable).
    Each process loads config independently to avoid cross-process issues.
    """
    console = Console()
    config = load_config(config_path)
    runner = SWERunner(output_dir)
    return runner.run_swe_task(task_row, config, console)

def to_abs(path: str, base: str | None = None) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    base_path = Path(base) if base else Path.cwd()
    return str((base_path / p).resolve())

def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench tasks (parallel)")
    parser.add_argument("--config", default="config/default.yaml", help="Agent configuration file")
    parser.add_argument(
        "--output_dir",
        default="./swe_outputs",
        help="Output directory for results",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 1),
                        help="Number of parallel workers (processes)")
    args = parser.parse_args()
    args.output_dir = to_abs(args.output_dir)


    console = Console()

    # load once in main process
    config = load_config(args.config)
    task_data = load_dataset("parquet", data_files={"test": config.parquet_path})
    rows = list(task_data["test"])

    console.print(f"Total tasks: {len(rows)}", style="bold cyan")
    console.print(f"Parallel workers: {args.workers}", style="bold cyan")

    all_results: List[Dict[str, Any]] = []
    success_cnt = 0

    # Run tasks in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(run_one_task, dict(row), args.config, args.output_dir): i
            for i, row in enumerate(rows)
        }

        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results = fut.result()
            except Exception as e:
                # if the process crashes unexpectedly
                results = {"success": False, "error": str(e), "instance_id": rows[i].get("instance_id", f"idx_{i}")}

            all_results.append(results)
            if results.get("success"):
                success_cnt += 1

            console.print(f"\n{'='*60}", style="bold blue")
            console.print(
                f"FINAL RESULTS [{len(all_results)}/{len(rows)}] - {results.get('instance_id')}",
                style="bold cyan",
            )
            console.print(f"{'='*60}", style="bold blue")
            console.print(
                f"Success: {results.get('success')}",
                style="bold green" if results.get("success") else "bold red",
            )
            if "evaluation" in results:
                eval_data = results["evaluation"]
                console.print(
                    f"FAIL_TO_PASS: {eval_data['fail_to_pass']['success']}",
                    style="green" if eval_data["fail_to_pass"]["success"] else "red",
                )
                console.print(
                    f"PASS_TO_PASS: {eval_data['pass_to_pass']['success']}",
                    style="green" if eval_data["pass_to_pass"]["success"] else "red",
                )

    # Save global summary
    summary = {
        "total": len(all_results),
        "success": success_cnt,
        "failed": len(all_results) - success_cnt,
        "results": all_results,
    }
    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    console.print(f"\n{'='*60}", style="bold blue")
    console.print("ALL TASKS SUMMARY", style="bold cyan")
    console.print(f"{'='*60}", style="bold blue")
    console.print(f"Total: {summary['total']}")
    console.print(f"Success: {summary['success']}", style="bold green" if summary["success"] else "bold")
    console.print(f"Failed: {summary['failed']}", style="bold red" if summary["failed"] else "bold green")
    console.print(f"Summary saved to: {summary_path}", style="blue")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())