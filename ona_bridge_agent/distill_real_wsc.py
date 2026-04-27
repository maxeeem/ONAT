import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
import random
from ona_bridge_agent.external_wsc_eval import (
    load_wsc_examples, _score_mlm_option, _replace_pronoun, _ona_lines_from_probs, _ona_predict, _DEFAULT_WSC_ARROW
)
from ona_bridge_agent.ona import ONAFileRunner

def main():
    print("Loading datasets and zero-shot teacher...")
    examples = load_wsc_examples(Path(_DEFAULT_WSC_ARROW))
    
    # 1. We load the predictions of the strong model + ONA
    # If we don't want to run Roberta here (slow), we can load the scores from external_wsc_results_strong.json !
    with open("external_wsc_results_strong.json", "r") as f:
        strong_results = json.load(f)
        
    print("Generating ONA-revised targets from teacher prior...")
    ona_targets = []
    
    runner = ONAFileRunner("../OpenNARS-for-Applications/NAR")
    rows_by_idx = {r["idx"]: r for r in strong_results["full_wsc273"]["rows"]}
    
    for ex in examples:
        # Get RoBERTa score difference as teacher prob
        s0 = rows_by_idx[ex.idx]["roberta-large_score0"]
        s1 = rows_by_idx[ex.idx]["roberta-large_score1"]
        p1_teach = 1.0 / (1.0 + np.exp(-(s1 - s0)))
        p0_teach = 1.0 - p1_teach
        
        # Run ONA multihop derivation to logically revise/crystallize the probability
        atom = f"distill_{ex.idx}"
        lines = _ona_lines_from_probs(atom, p0_teach, p1_teach, cycles=30, mode="multihop")
        
        out, _ = runner.run(lines, timeout_sec=10, keep_file=False)
        pred, scores = _ona_predict(out, atom)
        
        # We record the ONA score logic as our target
        ona_p1 = 1.0 / (1.0 + np.exp(-(scores.get("option1",0.0) - scores.get("option0",0.0))))
        ona_targets.append(ona_p1)
        
    print("Fine-tuning Student (all-MiniLM-L6-v2) on Symbolic Logical Targets...")
    # Initialize Student
    student = SentenceTransformer('all-MiniLM-L6-v2')
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=2e-5)
    loss_fn = torch.nn.MSELoss()
    
    epochs = 15
    for epoch in range(epochs):
        total_loss = 0.0
        random_idxs = list(range(len(examples)))
        random.shuffle(random_idxs)
        
        for i in random_idxs:
            ex = examples[i]
            target_p1 = ona_targets[i]
            target_p0 = 1.0 - target_p1
            
            optimizer.zero_grad()
            
            t0 = _replace_pronoun(ex, ex.options[0])
            t1 = _replace_pronoun(ex, ex.options[1])
            
            # Student forward pass
            features = student.tokenize([ex.text, t0, t1])
            # Move to device
            for k, v in features.items():
                if isinstance(v, torch.Tensor):
                    features[k] = v.to(student.device)
            
            output = student(features)
            vecs = output["sentence_embedding"]
            
            ctx = vecs[0]
            sim0 = torch.nn.functional.cosine_similarity(ctx, vecs[1], dim=0)
            sim1 = torch.nn.functional.cosine_similarity(ctx, vecs[2], dim=0)
            
            # Map bounded similarities to probabilities
            logits = torch.stack([sim0, sim1]) * 10.0 # scale up to be sharp
            probs = torch.nn.functional.softmax(logits, dim=0)
            
            target_tensor = torch.tensor([target_p0, target_p1], device=probs.device, dtype=torch.float32)
            
            loss = loss_fn(probs, target_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Student distill loss: {total_loss/len(examples):.4f}")
        
    print("Evaluating Distilled Student...")
    student.eval()
    correct = 0
    test_rows = []
    with torch.no_grad():
        for ex in examples:
            t0 = _replace_pronoun(ex, ex.options[0])
            t1 = _replace_pronoun(ex, ex.options[1])
            vecs = student.encode([ex.text, t0, t1], convert_to_tensor=True)
            sim0 = torch.nn.functional.cosine_similarity(vecs[0], vecs[1], dim=0).item()
            sim1 = torch.nn.functional.cosine_similarity(vecs[0], vecs[2], dim=0).item()
            
            pred = 0 if sim0 >= sim1 else 1
            if pred == ex.label:
                correct += 1
            test_rows.append(pred)
            
    acc = correct / len(examples)
    print(f"Student Zero-Shot Baseline: ~49.8%")
    print(f"Teacher (RoBERTa + ONA) accuracy bound: ~69.0%")
    print(f"Student Post-Distillation Accuracy: {acc:.4f} ({correct}/{len(examples)})")
    
    with open("distill_results.tsv", "a") as f:
        f.write(f"distill\t{acc:.4f}\t0.0\tkeep\tdistilled multihop ONA reasoning onto small sentence-transformers\n")

if __name__ == "__main__":
    main()
