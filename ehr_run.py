#!/usr/bin/env python3
"""
EHRSQL task runner that executes SafeFlow agents per question and evaluates answers.

Workflow per task:
1. Load EHRSQL JSON data (train/valid).
2. Filter out is_impossible = True entries.
3. For each id, launch a SafeFlow agent with (db_id, question) only.
4. Agent queries the corresponding SQLite DB to answer the question.
5. Evaluate agent output against ground-truth SQL query result.
6. Save results as CSV and report overall accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rich.console import Console

from agent.default import DefaultAgent
from utils import load_config

console = Console()


def to_abs(path: str, base: Optional[str] = None) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    base_path = Path(base) if base else Path.cwd()
    return str((base_path / p).resolve())


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        console.print(f"[yellow]Warning[/yellow]: data file not found: {path}")
        return []

    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def run_sql_query(db_path: Path, query: str) -> List[Tuple[Any, ...]]:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()


def normalize_query_result(rows: List[Tuple[Any, ...]]) -> Any:
    if not rows:
        return []
    if len(rows[0]) == 1:
        return [row[0] for row in rows]
    return [list(row) for row in rows]


def normalize_agent_answer(answer_text: str) -> Any:
    cleaned = answer_text.strip()
    if not cleaned:
        return ""
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return cleaned


def canonicalize(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def is_answer_correct(ground_truth: Any, agent_answer: Any) -> bool:
    gt_string = canonicalize(ground_truth)
    agent_string = canonicalize(agent_answer)
    if gt_string == agent_string:
        return True
    if isinstance(ground_truth, list) and len(ground_truth) == 1:
        return canonicalize(ground_truth[0]) == agent_string
    return False


def build_agent_prompt(question: str, db_id: str, db_path: Path) -> str:
    return (
        "You are a SafeFlow agent answering clinical data questions.\n\n"
        f"Database ID: {db_id}\n"
        f"Database path: {db_path}\n"
        f"Question: {question}\n\n"
        "Task:\n"
        "1. Use Python or sqlite to query the SQLite database above.\n"
        "2. Derive the answer from the database only.\n"
        "3. When finished, call base_tools__finish_task with verify_task=false.\n"
        "4. Put ONLY the final answer (or list of answers) in the finish_task message.\n"
    )


def extract_finish_message(run_result: Dict[str, Any]) -> str:
    messages = run_result.get("messages", [])
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        if message.get("name") != "base_tools__finish_task":
            continue
        content = message.get("content", "")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return content
        if isinstance(payload, dict):
            result = payload.get("result", {})
            if isinstance(result, dict) and "message" in result:
                return str(result["message"])
        return content
    return ""


def run_ehr_task(
    record: Dict[str, Any],
    config: Any,
    output_dir: Path,
    data_dir: Path,
) -> Dict[str, Any]:
    item_id = record.get("id", "unknown")
    question = record.get("question", "")
    db_id = record.get("db_id", "")
    query = record.get("query", "")

    task_dir = output_dir / item_id
    task_dir.mkdir(parents=True, exist_ok=True)

    db_path = data_dir / f"{db_id}.db"
    if not db_path.exists():
        console.print(f"[red]Missing DB[/red]: {db_path}")
        return {
            "id": item_id,
            "question": question,
            "ground_truth": "",
            "agent_answer": "",
            "correct": False,
            "error": f"DB not found: {db_path}",
        }

    try:
        query_rows = run_sql_query(db_path, query)
        ground_truth = normalize_query_result(query_rows)
    except Exception as exc:
        console.print(f"[red]Query failed[/red] for {item_id}: {exc}")
        return {
            "id": item_id,
            "question": question,
            "ground_truth": "",
            "agent_answer": "",
            "correct": False,
            "error": f"Query failed: {exc}",
        }

    prompt = build_agent_prompt(question=question, db_id=db_id, db_path=db_path)

    agent = DefaultAgent(
        config=config,
        item_id=item_id,
        work_root=str(task_dir),
    )

    try:
        run_result = agent.run(prompt)
        agent_message = extract_finish_message(run_result)
    except Exception as exc:
        console.print(f"[red]Agent failed[/red] for {item_id}: {exc}")
        agent_message = ""

    agent_answer = normalize_agent_answer(agent_message)
    correct = is_answer_correct(ground_truth, agent_answer)

    return {
        "id": item_id,
        "question": question,
        "ground_truth": ground_truth,
        "agent_answer": agent_answer,
        "correct": correct,
        "error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EHRSQL tasks with SafeFlow agents")
    parser.add_argument("--config", default="config/default.yaml", help="Agent configuration file")
    parser.add_argument("--output_dir", default="./ehr_outputs", help="Output directory for results")
    parser.add_argument("--data_dir", default="./data/ehrsql", help="Directory with EHRSQL DB/JSON files")
    parser.add_argument("--train_json", default="train.json", help="Training JSON filename")
    parser.add_argument("--valid_json", default="valid_json", help="Validation JSON filename")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records")

    args = parser.parse_args()
    output_dir = Path(to_abs(args.output_dir))
    data_dir = Path(to_abs(args.data_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)

    train_records = load_json_records(data_dir / args.train_json)
    valid_records = load_json_records(data_dir / args.valid_json)
    records = train_records + valid_records

    filtered_records = [r for r in records if not r.get("is_impossible", False)]
    if args.limit is not None:
        filtered_records = filtered_records[: args.limit]

    if not filtered_records:
        console.print("[yellow]No records found to process.[/yellow]")
        return

    results: List[Dict[str, Any]] = []
    correct_count = 0

    for idx, record in enumerate(filtered_records, start=1):
        item_id = record.get("id", "unknown")
        console.print(f"\n[{idx}/{len(filtered_records)}] Processing {item_id}...", style="cyan")
        result = run_ehr_task(record, config, output_dir, data_dir)
        results.append(result)
        if result.get("correct"):
            correct_count += 1

    accuracy = correct_count / len(filtered_records)

    results_csv = output_dir / "ehr_results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "question", "query_result", "agent_result"])
        for result in results:
            writer.writerow(
                [
                    result.get("id", ""),
                    result.get("question", ""),
                    canonicalize(result.get("ground_truth", "")),
                    canonicalize(result.get("agent_answer", "")),
                ]
            )

    summary_path = output_dir / "summary.json"
    summary = {
        "total": len(filtered_records),
        "correct": correct_count,
        "accuracy": accuracy,
        "results_csv": str(results_csv),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print("\nRun completed.", style="green")
    console.print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(filtered_records)})", style="green")
    console.print(f"Results saved to: {results_csv}", style="blue")


if __name__ == "__main__":
    main()
