import torch
from model import ToyTensionModel

torch.autograd.set_detect_anomaly(True)

def test_anomaly():
    print("Initializing Float32 RMN Model...")
    model = ToyTensionModel(vocab_size=1000, d_model=128, depth=2).to(torch.float32)
    model.train()
    
    # 2 samples, 8 tokens
    input_ids = torch.randint(0, 1000, (2, 8))
    target_ids = torch.randint(0, 1000, (2, 8))
    
    print("Executing Forward Pass...")
    logits = model(input_ids)
    
    # Check forward nan
    if torch.isnan(logits).any():
        print("FORWARD PASS PRODUCED NAN!")
    
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits.view(-1, 1000), target_ids.view(-1))
    
    print(f"Forward Loss: {loss.item()}")
    
    print("Executing Backward Pass...")
    loss.backward()
    print("Backward pass completed successfully.")

if __name__ == "__main__":
    test_anomaly()
