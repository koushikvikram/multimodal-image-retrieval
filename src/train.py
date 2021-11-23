'''Training, Validation and Testing functions'''
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


def encode(test_loader, model):
    '''get image embeddings from the model'''
    test_image_embeddings = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.to(device)
        model.eval()
        for i, (image_names, images) in enumerate(test_loader):
            print(f"Processing Batch {i}/{len(test_loader)}")
            images = images.to(device)
            predictions = model(images)
            for batch_idx, embedding in enumerate(predictions):
                image_ID = image_names[batch_idx]
                embedding = embedding.cpu().detach().numpy()
                test_image_embeddings[image_ID] = embedding
    return test_image_embeddings


def save_checkpoint(model, filename):
    '''save model to disk'''
    torch.save(model.state_dict(), filename+".pth.tar")
