'''Training and Validation functions'''
import torch


def train(train_loader, model, criterion, optimizer):
    '''trains model for a single epoch'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    running_loss = 0.0 # initialize loss for entire epoch
    for i, (image_names, images, embeddings) in enumerate(train_loader):
        images, embeddings = images.to(device), embeddings.to(device)
        optimizer.zero_grad()
        # get prediction for batch and compute loss
        outputs = model(images)
        loss = criterion(outputs, embeddings)
        # compute gradient for batch and backpropagate
        loss.backward()
        optimizer.step()
        # update running_loss for epoch
        running_loss += loss.item()
        # print loss for each batch
        print(f"train - epoch: {epoch} \t\t batch: {i+1}/{len(train_loader)} \t\t average batch loss: {loss/len(images)}")
    avg_loss = running_loss/len(train_loader.dataset)
    return avg_loss


def validate(val_loader, model, criterion):
    '''validate trained model'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.to(device)
        model.eval()
        running_loss = 0.0 # initialize loss for entire epoch
        for i, (image_names, images, embeddings) in enumerate(val_loader):
            images, embeddings = images.to(device), embeddings.to(device)
            # get prediction for batch and compute loss
            outputs = model(images)
            loss = criterion(outputs, embeddings)
            # update running_loss for epoch
            running_loss += loss.item()
            # print loss for each batch
            print(f"val - epoch: {epoch} \t\t batch: {i+1}/{len(val_loader)} \t\t average batch loss: {loss/len(images)}")
        avg_loss = running_loss/len(val_loader.dataset)
    return avg_loss


def save_checkpoint(model, filename):
    '''save model to disk'''
    torch.save(model.state_dict(), filename+".pth.tar")
