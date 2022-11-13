import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from tqdm import tqdm

from model import GoogLeNet

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./runs')

def main(resume=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_transform = {
        'train': transforms.Compose(
            [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        ),
        'test': transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )
    }

    
    train_dataset = datasets.CIFAR10('./', train=True, transform=data_transform['train'], download=True)
    test_dataset = datasets.CIFAR10('./', train=False, transform=data_transform['test'], download=True)
    
    batch_size = 128
    num_workers = min([os.cpu_count(),  8])
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size, num_workers=num_workers)
    
    train_num, test_num = len(train_dataset), len(test_dataset)
    
    net = GoogLeNet(num_classes=10, aux_logits=False)
    net.to(device)
    

    epochs = 50

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_acc = 0.0
    save_path = './googleNet.pth'
    train_steps = len(train_loader)
    
    if resume:
        checkpoint = torch.load(save_path, map_location=device)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        print(f'-----恢复上次训练, 上次准确率为{best_acc}-----')

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        acc = 0
        
        train_bar = tqdm(train_loader, file=sys.stdout)
        for i, (X, y) in enumerate(train_bar):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            if net.aux_logits:
                logits, aux_logits2, aux_logits1 = net(X)
                loss0 = criterion(logits, y)
                loss1 = criterion(aux_logits1, y)
                loss2 = criterion(aux_logits2, y)
                loss = loss0 + 0.3 * loss1 + 0.3 * loss2
            else:
                logits = net(X)
                loss = criterion(logits, y)
            
            y_hat = torch.argmax(logits, dim=1)
            acc += (y_hat==y).sum().item()


            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            tmp_lr = optimizer.param_groups[0]['lr']

            train_bar.desc = f'train epoch[{epoch+1}/{epochs}] loss:{loss:.3f} lr:{tmp_lr}'
        
        train_acc = acc / train_num

        scheduler.step()

        net.eval()
        acc = 0
        
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            
            for X, y in test_bar:
                X, y = X.to(device), y.to(device)
                outputs = net(X)
                y_hat = torch.argmax(outputs, dim=1)
                
                acc += (y_hat==y).sum().item()
        
        test_acc = acc / test_num
        
        print(f'[epoch{epoch+1}/{epochs}] train_loss: {running_loss / train_steps: .4f}  train_accuracy: {train_acc: .4f}  test_accuracy: {test_acc: .4f}')
        writer.add_scalar('train_loss', running_loss / train_steps, epoch+1)
        writer.add_scalar('train_accuracy', train_acc, epoch+1)
        writer.add_scalar('test_accuracy', test_acc, epoch+1)

        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'net': net.state_dict(),
                'acc': best_acc,
                'epoch': epoch
            }
            torch.save(state, save_path)

if __name__ == '__main__':
    main()
