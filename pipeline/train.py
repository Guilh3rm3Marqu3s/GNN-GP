import torch



def train_one_epoch(model, optimizer, data, criterion, set='train'):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    
    if set == 'train':
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
    elif set == 'train_val':
        combined_mask = torch.logical_or(data.train_mask, data.val_mask)
        loss = criterion(out[combined_mask], data.y[combined_mask])
    
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, mask=None):
    
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        
        if mask is None:
            mask = data.test_mask
        
        correct = pred[mask] == data.y[mask]
        
        acc = int(correct.sum()) / int(mask.sum())
        
        return acc