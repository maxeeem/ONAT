from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path


_TRUTH_RE = re.compile(r"%\s*([0-9.]+)\s*;\s*([0-9.]+)\s*%")


class ONARuntimeError(RuntimeError):
    pass


class ONAFileRunner:
    """
    Runs ONA in file-stream mode:

        ./NAR shell < generated.nal

    This is more reliable than interactive stdin/stdout for smoke tests because
    ONA's shell output/flushing differs by build/platform.
    """

    def __init__(self, ona_cmd: str):
        self.ona_cmd = ona_cmd

    def run(self, lines: list[str], timeout_sec: int = 10, keep_file: bool = False) -> tuple[str, str | None]:
        with tempfile.NamedTemporaryFile("w", suffix=".nal", delete=False, encoding="utf-8") as f:
            for line in lines:
                f.write(line.rstrip() + "\n")
            path = f.name

        cmd = self.ona_cmd.split() + ["shell"]
        try:
            with open(path, "r", encoding="utf-8") as stdin:
                proc = subprocess.run(
                    cmd,
                    stdin=stdin,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_sec,
                )
        except FileNotFoundError as exc:
            raise ONARuntimeError(f"Could not find ONA command: {self.ona_cmd!r}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ONARuntimeError(f"ONA timed out after {timeout_sec}s") from exc
        finally:
            if not keep_file:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass

        if proc.returncode != 0:
            raise ONARuntimeError(f"ONA exited with code {proc.returncode}\n{proc.stdout}")
        return proc.stdout, path if keep_file else None


def truth_score(line: str) -> float:
    """Score a line by truth expectation-ish f*c; fallback to 1 for hits."""
    m = _TRUTH_RE.search(line)
    if not m:
        return 1.0
    return float(m.group(1)) * float(m.group(2))


def max_score_for_term(output: str, term: str) -> float:
    term_lower = term.lower()
    best = 0.0
    for line in output.splitlines():
        low = line.lower()
        if term_lower in low:
            # Ignore raw input echo lines if possible. ONA often prefixes derived output.
            if low.strip().startswith("input") or low.strip().startswith("//"):
                continue
            best = max(best, truth_score(low))
    return best


def predict_from_ona_output(output: str, adjective: str) -> tuple[str | None, dict[str, float]]:
    subj_term = f"<{adjective} --> subject_cause_of_fit_failure>"
    obj_term = f"<{adjective} --> object_cause_of_fit_failure>"
    scores = {
        "subject": max_score_for_term(output, subj_term),
        "object": max_score_for_term(output, obj_term),
    }
    if scores["subject"] == 0 and scores["object"] == 0:
        return None, scores
    if scores["subject"] >= scores["object"]:
        return "subject", scores
    return "object", scores
