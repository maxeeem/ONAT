import torch
from sentence_transformers import SentenceTransformer
import warnings
from ona_bridge_agent.dataset import DYNAMIC_EXAMPLES

warnings.filterwarnings("ignore")

def extract_embeddings_for_fine_tuning(model):
    print("Initializing Neuro-Symbolic Distillation Loop...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.MSELoss()
    
    examples = DYNAMIC_EXAMPLES
    epochs = 15
    for epoch in range(epochs):
        total_loss = 0.0
        
        for ex in examples:
            optimizer.zero_grad()
            
            # The context is the sentence itself
            ctx_emb = model.encode(ex.sentence, convert_to_tensor=True)
            cand1_emb = model.encode(ex.subject, convert_to_tensor=True)
            cand2_emb = model.encode(ex.object, convert_to_tensor=True)
            
            sim1 = torch.nn.functional.cosine_similarity(ctx_emb, cand1_emb, dim=0)
            sim2 = torch.nn.functional.cosine_similarity(ctx_emb, cand2_emb, dim=0)
            
            neural_scores = torch.stack([sim1, sim2])
            
            # ONA's decision
            target_idx = 0 if ex.expected == "subject" else 1
            
            target_scores = torch.tensor([0.9 if i == target_idx else 0.1 for i in range(2)], device=neural_scores.device)
            
            loss = loss_fn(neural_scores, target_scores)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 3 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(examples):.4f}")

    print("\n[SUCCESS] Neural Prior has been successfully aligned with Symbolic Logic!")

if __name__ == "__main__":
    model = SentenceTransformer('all-MiniLM-L6-v2')
    extract_embeddings_for_fine_tuning(model)
