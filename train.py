import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from data import load_data
from model import FineGrainedClassifier

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        train_acc += (predicted == target).sum().item()

    train_loss /= len(train_loader)
    train_acc /= len(train_dataset)

    return train_loss, train_acc

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            test_acc += (predicted == target).sum().item()

    test_loss /= len(test_loader)
    test_acc /= len(test_dataset)

    return test_loss, test_acc

if __name__ == '__main__':
    data_dir = "C:\fine-grained-classifier"
    batch_size = 32
    num_epochs = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FineGrainedClassifier(num_classes=num_artists).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_loader, test_loader = load_data(data_dir, batch_size)

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}')
        print(f'Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}')

    torch.save(model.state_dict(), 'fine-grained-classifier.pth')