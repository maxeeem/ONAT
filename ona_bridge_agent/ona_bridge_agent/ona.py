from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path


_TRUTH_RE = re.compile(r"truth:\s*frequency=([0-9.]+),\s*confidence=([0-9.]+)", re.IGNORECASE)


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
    """Score a line by truth expectation-ish f*c."""
    m = _TRUTH_RE.search(line)
    if not m:
        return 0.0
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


def predict_from_ona_output(output: str, adjective: str) -> tuple[str | None, dict[str, float], dict[str, list[str]]]:
    subj_term = f"<{adjective} --> subject_cause_of_fit_failure>"
    obj_term = f"<{adjective} --> object_cause_of_fit_failure>"
    scores = {
        "subject": max_score_for_term(output, subj_term),
        "object": max_score_for_term(output, obj_term),
    }
    
    # Extract explanations via stamps
    explanations = {"subject": [], "object": []}
    explanations["subject"] = extract_explanation(output, subj_term)
    explanations["object"] = extract_explanation(output, obj_term)
    
    if scores["subject"] == 0 and scores["object"] == 0:
        return None, scores, explanations
    if scores["subject"] >= scores["object"]:
        return "subject", scores, explanations
    return "object", scores, explanations


def extract_explanation(output: str, term: str) -> list[str]:
    """Strict chronological reconstruction of the NAL derivation chain."""
    term_lower = term.lower()
    
    line_re = re.compile(r"^(Input|Derived|Answer|Selected):\s*(.+?)\.\s*Priority=.*?Stamp=\[([0-9,]+)\]", re.IGNORECASE)
    ans_re = re.compile(rf"^(Answer|Derived):\s*{re.escape(term_lower)}\.", re.IGNORECASE)
    
    target_stamp_set = set()
    best_conf = -1.0
    
    # 1. Find the best confidence for the target term to identify the winning derivation stamp
    for line in output.splitlines():
        line = line.strip()
        if ans_re.match(line):
            sc = truth_score(line)
            if sc > best_conf:
                m = line_re.match(line)
                if m:
                    best_conf = sc
                    target_stamp_set = set(m.group(3).split(","))
                    
    if not target_stamp_set:
        return []
        
    # 2. Re-parse and collect the causal chronological chain
    # Only keep lines whose stamps are subsets of the target_stamp_set.
    explanation_chain = []
    seen_statements = set()
    
    for line in output.splitlines():
        line = line.strip()
        m = line_re.match(line)
        if not m:
            continue
            
        line_type = m.group(1).capitalize()
        statement = m.group(2).strip()
        stamp_str = m.group(3)
        stamp_set = set(stamp_str.split(","))
        
        # If the stamp is a subset of the target stamp, it contributed to the final answer.
        # We also ignore Answers since they don't add to the derivation steps.
        if line_type != "Answer" and stamp_set.issubset(target_stamp_set):
            # Only include basic Inheritance/Implication derivations to keep it interpretable and minimal.
            # Exclude Symmetric (A <-> B) and complex higher-order variables that ONA spams unless they are Inputs!
            if ("<->" in statement or "$1" in statement):
                continue
                
            formatted_step = f"{statement} (Stamp: {stamp_str})"
            if formatted_step not in seen_statements:
                # Deduplicate transitive inverses (A->B and B->A) if not strictly necessary, 
                # but ONA needs to print them so let's just use the main chain.
                explanation_chain.append(f"{line_type}: {formatted_step}")
                seen_statements.add(formatted_step)
                
    return explanation_chain
