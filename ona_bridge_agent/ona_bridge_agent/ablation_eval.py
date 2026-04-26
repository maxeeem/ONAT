import json
import numpy as np
from .bridge import SentenceTransformerConceptEmbedder, FitReasoningBridge
from .types import Example
from .ona import ONAFileRunner, predict_from_ona_output

def main():
    print("Running Rigorous Confidence Ablation Study...")
    embedder = SentenceTransformerConceptEmbedder("all-MiniLM-L6-v2")
    bridge = FitReasoningBridge(embedder=embedder, concept_threshold=0.20)
    runner = ONAFileRunner("../OpenNARS-for-Applications/NAR")
    
    ex = Example(
        sentence="The trophy did not fit in the suitcase because it was large.",
        subject="trophy",
        object="suitcase",
        adjective="large",
        expected="object"
    )
    
    frame = bridge.extract(ex.sentence, known_adjective=ex.adjective)
    base_narsese = bridge.to_narsese(frame, cycles=60) # more cycles for deeper deduction
    
    results = []
    
    for conf in np.arange(0.1, 1.0, 0.1):
        conf_val = round(conf, 2)
        # Direct contradiction for ablation:
        # Instead of suitcase=magic_bag->small, let's inject `<large --> small_like>. %1.00;conf%` properly
        # so it directly competes with the embedding prior `<large --> large_like>`
        conflict_rules = [
            f"<large --> small_like>. %1.00;{conf_val:.2f}%"
        ]
        
        narsese = base_narsese[:-3] + conflict_rules + base_narsese[-3:]
        
        ona_output, _ = runner.run(narsese, timeout_sec=10, keep_file=False)
        pred, scores, _ = predict_from_ona_output(ona_output, ex.adjective)
        
        results.append({
            "injected_confidence": conf_val,
            "subject_score": scores.get("subject", 0.0),
            "object_score": scores.get("object", 0.0),
            "prediction": pred
        })
        print(f"Conf: {conf_val:.2f} -> Pred: {pred} (Subj: {scores.get('subject',0):.3f}, Obj: {scores.get('object',0):.3f})")
        
    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
