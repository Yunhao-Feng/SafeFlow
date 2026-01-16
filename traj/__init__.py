from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json, shutil, logging


@dataclass
class TraceTrack:
    root_dir: Path            # 例如 "outputs/"
    item_id: str              # 每个数据项一个文件夹：outputs/{item_id}/
    run_name: str = "run"
    run_dir: Path = field(init=False)
    trace_file: Path = field(init=False)

    def __post_init__(self):
        safe_id = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in self.item_id)
        self.run_dir = Path(self.root_dir) / safe_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = self.run_dir / f"{self.run_name}_trace.txt"
        self.trace_file.touch(exist_ok=True)

    def _ts(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 追加写一条轨迹（带时间戳）
    def step(self, msg: str):
        prefix = f"[{self._ts()}]"
        with self.trace_file.open("a", encoding="utf-8") as f:
            f.write(prefix + " " + msg.rstrip() + "\n")

    # 保存文本/JSON到该数据项文件夹
    def save_text(self, name: str, text: str):
        p = self.run_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def save_json(self, name: str, obj):
        p = self.run_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return p
    # 拷贝生成的文件到该文件夹
    def copy_in(self, src_path: str | Path, dst_name: str | None = None):
        src = Path(src_path)
        dst = self.run_dir / (dst_name or src.name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst
        
    # 把 logging 输出也写入该数据项文件夹（例如 run.log）
    def attach_logging(self, level=logging.DEBUG, filename: str = "run.log"):
        log_path = self.run_dir / filename
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root = logging.getLogger()
        root.setLevel(level)
        root.addHandler(handler)
        return log_path