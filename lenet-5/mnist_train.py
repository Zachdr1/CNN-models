import torch.optim as optim
import torch.nn as nn
import torch
from mnist_data import load_data
from lenet5 import Net

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=.001, momentum=.9)
    trainloader = load_data()
    dataiter = iter(trainloader)

    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(trainloader,0):
            inputs , labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs) # forward
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print metrics
            running_loss += loss.item()
            print(f'Epoch: {epoch}\t Loss: {running_loss}')
            running_loss = 0.0
    print('Finished Training')

if __name__ == '__main__':
    main()


