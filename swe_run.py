#!/usr/bin/env python3
"""
SWE-bench task runner that handles repository preparation and agent execution.

This script handles the SWE workflow:
1. Clone repository and checkout to base_commit
2. Apply test_patch to expose the bug
3. Prepare task description for agent (without leaking patch)
4. Run agent to fix the issue
5. Evaluate final results
"""

import json
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse
from datasets import load_dataset
from rich.console import Console
from agent.default import DefaultAgent
from utils import load_config

console = Console()


def to_abs(path: str, base: str | None = None) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    base_path = Path(base) if base else Path.cwd()
    return str((base_path / p).resolve())

class SWERunner:
    """Handles SWE-bench task execution workflow."""

    def __init__(self, output_dir: str = "./swe_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def prepare_repository(self, task_data: Dict[str, Any], work_dir: Path) -> bool:
        """
        Prepare repository for SWE task.

        Args:
            task_data: SWE task data containing repo, base_commit, etc.
            work_dir: Directory to clone repository into

        Returns:
            bool: True if preparation successful
        """
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
            except:
                console.print(f"Cloning {repo_url}...", style="cyan")
                subprocess.run([
                    "git", "clone", repo_url, str(repo_path)
                ], check=True, capture_output=True)

            # 2. Checkout to base_commit (task starting point)
            console.print(f"Checking out base commit {task_data['base_commit']}...", style="cyan")
            subprocess.run([
                "git", "checkout", task_data['base_commit']
            ], cwd=str(repo_path), check=True, capture_output=True)

            # 3. Apply test_patch to expose the bug
            if task_data.get('test_patch'):
                console.print("Applying test patch...", style="cyan")
                test_patch_file = work_dir / "test_patch.diff"
                test_patch_file.write_text(task_data['test_patch'])

                result = subprocess.run([
                    "git", "apply", str(test_patch_file)
                ], cwd=str(repo_path), capture_output=True)

                if result.returncode != 0:
                    console.print(f"[yellow]Warning[/yellow]: Test patch application failed: {result.stderr.decode()}")
                    # Try with patch command as fallback
                    result = subprocess.run([
                        "patch", "-p1", "-i", str(test_patch_file)
                    ], cwd=str(repo_path), capture_output=True)

                    if result.returncode != 0:
                        console.print("[red]Error[/red]: Both git apply and patch failed")
                        return False

            return True

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error preparing repository[/red]: {e}")
            return False

    def run_tests(self, repo_path: Path, test_commands: List[str]) -> Tuple[bool, str]:
        """
        Run specified test commands and return results.

        Args:
            repo_path: Path to repository
            test_commands: List of test commands to run

        Returns:
            Tuple of (success, output)
        """
        all_output = []
        all_passed = True

        for cmd in test_commands:
            try:
                # Run pytest command
                result = subprocess.run([
                    "python", "-m", "pytest", cmd, "-v"
                ], cwd=str(repo_path), capture_output=True, text=True, timeout=120)

                output = f"Command: {cmd}\n"
                output += f"Return code: {result.returncode}\n"
                output += f"STDOUT:\n{result.stdout}\n"
                output += f"STDERR:\n{result.stderr}\n"
                output += "-" * 80 + "\n"

                all_output.append(output)

                if result.returncode != 0:
                    all_passed = False

            except subprocess.TimeoutExpired:
                output = f"Command: {cmd}\nTIMEOUT after 120 seconds\n"
                all_output.append(output)
                all_passed = False

        return all_passed, "\n".join(all_output)

    def evaluate_solution(self, repo_path: Path, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the agent's solution against FAIL_TO_PASS and PASS_TO_PASS tests.

        Args:
            repo_path: Path to repository with agent's changes
            task_data: Original task data with test lists

        Returns:
            Evaluation results
        """
        results = {
            "fail_to_pass": {"tests": [], "success": False, "output": ""},
            "pass_to_pass": {"tests": [], "success": False, "output": ""}
        }

        # Parse test lists from string format
        try:
            fail_to_pass = json.loads(task_data.get('FAIL_TO_PASS', '[]'))
            pass_to_pass = json.loads(task_data.get('PASS_TO_PASS', '[]'))
        except json.JSONDecodeError:
            console.print("[red]Error parsing test lists[/red]")
            return results

        # Run FAIL_TO_PASS tests (should now pass)
        if fail_to_pass:
            console.print(f"Running {len(fail_to_pass)} FAIL_TO_PASS tests...", style="cyan")
            success, output = self.run_tests(repo_path, fail_to_pass)
            results["fail_to_pass"] = {
                "tests": fail_to_pass,
                "success": success,
                "output": output
            }

        # Run PASS_TO_PASS tests (should continue to pass)
        if pass_to_pass:
            console.print(f"Running {len(pass_to_pass)} PASS_TO_PASS tests...", style="cyan")
            success, output = self.run_tests(repo_path, pass_to_pass)
            results["pass_to_pass"] = {
                "tests": pass_to_pass,
                "success": success,
                "output": output
            }

        return results

    def prepare_agent_task(self, task_data: Dict[str, Any], repo_path: Path) -> str:
        """
        Prepare task description for agent (without leaking the patch solution).

        Args:
            task_data: Original SWE task data
            repo_path: Path to prepared repository

        Returns:
            Task description for agent
        """
        # IMPORTANT: Do not include the 'patch' field - that's the solution!
        safe_task_data = {
            "repo": task_data["repo"],
            "instance_id": task_data["instance_id"],
            "problem_statement": task_data["problem_statement"],
            "repository_path": str(repo_path),
            "fail_to_pass_tests": json.loads(task_data.get('FAIL_TO_PASS', '[]')),
            "pass_to_pass_tests": json.loads(task_data.get('PASS_TO_PASS', '[]')),
            "hints": task_data.get("hints_text", ""),
            "difficulty": task_data.get("difficulty", "unknown")
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

    def run_swe_task(self, task_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete SWE task workflow.

        Args:
            task_data: SWE task data
            config: Agent configuration

        Returns:
            Results including success status and evaluation
        """
        instance_id = task_data['instance_id']
        console.print(f"\n{'='*60}", style="bold blue")
        console.print(f"Running SWE Task: {instance_id}", style="bold cyan")
        console.print(f"Repository: {task_data['repo']}", style="cyan")
        console.print(f"{'='*60}", style="bold blue")

        # Create working directory for this task
        task_dir = self.output_dir / instance_id
        task_dir.mkdir(exist_ok=True)

        try:
            # 1. Prepare repository
            if not self.prepare_repository(task_data, task_dir):
                return {"success": False, "error": "Failed to prepare repository"}

            repo_name = task_data['repo'].split('/')[-1]
            repo_path = task_dir / repo_name

            # 2. Run initial tests to verify bug exists
            console.print("Verifying that FAIL_TO_PASS tests are currently failing...", style="cyan")
            fail_to_pass = json.loads(task_data.get('FAIL_TO_PASS', '[]'))
            if fail_to_pass:
                initial_success, initial_output = self.run_tests(repo_path, fail_to_pass)
                if initial_success:
                    console.print("[yellow]WARNING[/yellow]: FAIL_TO_PASS tests are already passing - bug may not be reproduced")

            # 3. Prepare agent task (without solution patch)
            agent_prompt = self.prepare_agent_task(task_data, repo_path)

            # 4. Run agent
            console.print("Starting agent to fix the issue...", style="green")

            # Initialize agent with working directory set to repo
            agent = DefaultAgent(
                config=config,
                item_id=instance_id,
                work_root=str(repo_path)
            )

            # Run agent with the prepared prompt
            try:
                agent.run(agent_prompt)
                agent_success = True
                agent_error = None
            except Exception as e:
                agent_success = False
                agent_error = str(e)
                console.print(f"[red]Agent execution failed[/red]: {e}")

            # 5. Evaluate results
            console.print("Evaluating final solution...", style="cyan")
            evaluation = self.evaluate_solution(repo_path, task_data)

            # Determine overall success
            overall_success = (
                agent_success and
                evaluation["fail_to_pass"]["success"] and
                evaluation["pass_to_pass"]["success"]
            )

            results = {
                "success": overall_success,
                "instance_id": instance_id,
                "repo": task_data['repo'],
                "agent_success": agent_success,
                "agent_error": agent_error,
                "evaluation": evaluation,
                "repository_path": str(repo_path)
            }

            # Save results
            results_file = task_dir / "results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            console.print(f"\nTask completed. Overall success: {overall_success}", style="bold green" if overall_success else "bold red")
            console.print(f"Results saved to: {results_file}", style="blue")

            return results

        except Exception as e:
            console.print(f"[red]Error running SWE task[/red]: {e}")
            return {
                "success": False,
                "error": str(e),
                "instance_id": instance_id
            }


def main():
    """Main entry point for SWE runner."""
    parser = argparse.ArgumentParser(description="Run SWE-bench tasks")
    parser.add_argument("--config", default="config/default.yaml", help="Agent configuration file")
    parser.add_argument("--output_dir", default="./swe_outputs", help="Output directory for results")

    args = parser.parse_args()
    args.output_dir = to_abs(args.output_dir)
    # Load configuration
    config = load_config(args.config)

    # Load task data
    task_data = load_dataset("parquet", data_files={"test": config.parquet_path})

    # Run SWE tasks (sequential)
    runner = SWERunner(args.output_dir)

    all_results = []
    success_cnt = 0

    for i, row in enumerate(task_data["test"]):
        results = runner.run_swe_task(row, config)
        all_results.append(results)
        if results.get("success"):
            success_cnt += 1

        # Print per-task final results
        console.print(f"\n{'='*60}", style="bold blue")
        console.print(f"FINAL RESULTS [{i+1}/{len(task_data['test'])}] - {results.get('instance_id')}", style="bold cyan")
        console.print(f"{'='*60}", style="bold blue")
        console.print(f"Success: {results.get('success')}", style="bold green" if results.get("success") else "bold red")
        if "evaluation" in results:
            eval_data = results["evaluation"]
            console.print(
                f"FAIL_TO_PASS: {eval_data['fail_to_pass']['success']}",
                style="green" if eval_data["fail_to_pass"]["success"] else "red"
            )
            console.print(
                f"PASS_TO_PASS: {eval_data['pass_to_pass']['success']}",
                style="green" if eval_data["pass_to_pass"]["success"] else "red"
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
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    console.print(f"\n{'='*60}", style="bold blue")
    console.print("ALL TASKS SUMMARY", style="bold cyan")
    console.print(f"{'='*60}", style="bold blue")
    console.print(f"Total: {summary['total']}")
    console.print(f"Success: {summary['success']}", style="bold green" if summary["success"] else "bold")
    console.print(f"Failed: {summary['failed']}", style="bold red" if summary["failed"] else "bold green")
    console.print(f"Summary saved to: {summary_path}", style="blue")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    exit(main())