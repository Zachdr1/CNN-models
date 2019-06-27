import torch
from mnist_data import load_data
from lenet5 import Net

def main():
    filename = input('Enter file name for model: ')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = torch.load(filename)
    _, testloader = load_data()
    # Test
    print('Starting Testing \n')
    correct = 0 
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct+= (predicted == labels).sum().item()
    print(f'Accuracy: {100*correct/total}')

if __name__ == '__main__':
    main()