import torch



def train_one_epoch(model, optimizer, data, criterion):
    model.train()
    optimizer.zero_grad()
    
    out = model(data)
    
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data):
    
    model.eval()
    
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        
        correct = pred[data.test_mask] == data.y[data.test_mask]
        
        acc = int(correct.sum()) / int(data.test_mask.sum())
        
        return acc