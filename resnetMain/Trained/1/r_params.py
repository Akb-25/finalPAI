from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import resnet as models
from torch.optim.lr_scheduler import StepLR

from perforatedai import pb_network as PN
from perforatedai import pb_globals as PBG
from perforatedai import pb_models as PBM

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0

    #Loop over all the batches in the dataset
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        #Pass the data through your model to get the output
        output = model(data)
        #Calculate the error
        loss = F.cross_entropy(output, target)
        #Backpropagate the error through the network
        loss.backward()
        #Modify the weights based on the calculated gradient
        optimizer.step()
        #Display Metrics
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        #Determine the predictions the network was making
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        #Increment how many times it was correct
        correct += pred.eq(target.view_as(pred)).sum()
    
    model.to(device)


def test(model, device, test_loader, optimizer, scheduler, args):
    model.eval()
    test_loss = 0
    correct = 0
    #Dont calculate Gradients
    with torch.no_grad():
        #Loop over all the test data
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #Pass the data through your model to get the output
            output = model(data)
            #Calculate the error
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            #Determine the predictions the network was making
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #Increment how many times it was correct
            correct += pred.eq(target.view_as(pred)).sum()

    #Display Metrics
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return model, optimizer, scheduler


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--save-name', type=str, default='PB')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    num_classes = 47
    image_size = 32
    #Define the data loaders
    transform_train = transforms.Compose(
            [ 
                #transforms.CenterCrop(26),
                transforms.Resize((image_size,image_size)),
                transforms.RandomRotation(10),      
                transforms.RandomAffine(5),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])
    transform_test = transforms.Compose(
            [ 
                transforms.Resize((image_size,image_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
        transforms.Normalize((0.1307,), (0.3081,)),
            ])
    #Dataset
    dataset1 = datasets.EMNIST(root='../../data', split='balanced', train=True, download=True, transform=transform_train)

    dataset2 = datasets.EMNIST(root='../../data',  split='balanced', train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    PBG.moduleNamesToConvert.append('BasicBlock')
    
    #Create the model
    # model = models.resnet18(num_classes == num_classes)
    #model = PBM.ResNetPB(model)
    

    
    # model = PN.loadPAIModel(model, 'PBOld/best_model_pai.pt')
    # model = PN.loadPAIModel(model, '125Model.pt')
    # model = model.to(device)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}")
    
    # model = PN.loadPAIModel(model, '25Model.pt')
    # model = model.to(device)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}")
   

    number = 8
    models = []
    for i in range(1, number+1):
        models.append(f"modelsInit1/{i}_pruned_model.pth")
    print(models)
    for loc in models:
        print(loc)
        model = torch.load(loc, weights_only=False)
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        
    sys.exit()
    
    
    #Setup the optimizer and scheduler
    optimArgs = {'params':model.parameters(),'lr':args.lr}
    optimizer = optim.Adadelta(**optimArgs)
    schedArgs = {'optimizer': optimizer, 'step_size':1, 'gamma': args.gamma}
    scheduler = StepLR(**schedArgs)


    #Run your epochs of training and testing
    for epoch in range(1, args.epochs + 1):
        
        #No training for my test, but you should add this back
        #train(args, model, device, train_loader, optimizer, epoch)
        
        
        model, optimizer, scheduler = test(model, device, test_loader, optimizer, scheduler, args)
        scheduler.step()
        
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
