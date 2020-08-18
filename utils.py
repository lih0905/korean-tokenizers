
import torch
from tqdm import tqdm

# calculate the elapsed time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# calculate accuracy
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds==y).float()
    acc = correct.sum() / len(correct)
    return acc

# train, evaluate functions
def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        #print_shape('predictions',predictions)
        
        loss = criterion(predictions, batch.label)
        #print_shape('loss',loss)
        
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(iterator):
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)#.squeeze(0))
            acc = binary_accuracy(predictions, batch.label)#.squeeze(0))

            epoch_loss += loss.item()
            epoch_acc += acc.item()
                    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
