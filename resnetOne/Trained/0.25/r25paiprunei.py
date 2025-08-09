from __future__ import print_function
import os
import pandas as pd
import argparse
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import resnet as models
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from perforatedai import pb_globals as PBG
from perforatedai import pb_models as PBM
from perforatedai import pb_utils as PBU


def prune_filters(model, indices):
      conv_layer=0
      device = torch.device("cuda")

      for layer_name, layer_module in model.named_modules():

        if(isinstance(layer_module, th.nn.Conv2d) and not layer_name.startswith('conv1')):

          if "conv1" in layer_name:
            in_channels=[i for i in range(layer_module.weight.shape[1])]
            out_channels = indices[conv_layer]
            layer_module.weight = th.nn.Parameter(
              th.FloatTensor(layer_module.weight.data.cpu().numpy()[out_channels]).to(device)
            )

          elif 'conv2' in layer_name:
            in_channels = indices[conv_layer]
            out_channels=[i for i in range(layer_module.weight.shape[0])]
            layer_module.weight = th.nn.Parameter(
                th.FloatTensor(layer_module.weight.data.cpu().numpy()[:, in_channels]).to(device)
            )
            conv_layer += 1

          layer_module.in_channels = len(in_channels)
          layer_module.out_channels = len(out_channels)

        if (isinstance(layer_module, th.nn.BatchNorm2d) and layer_name!='bn1' and layer_name.find('bn1')!=-1):
          out_channels=indices[conv_layer]
          layer_module.weight = th.nn.Parameter(
              th.FloatTensor(layer_module.weight.data.cpu().numpy()[out_channels]).to(device)
          )
          layer_module.bias = th.nn.Parameter(
            th.FloatTensor(layer_module.bias.data.cpu().numpy()[out_channels]).to(device)
          )
          layer_module.running_mean = th.FloatTensor(layer_module.running_mean.cpu().numpy()[out_channels]).to(device)
          layer_module.running_var = th.FloatTensor(layer_module.running_var.cpu().numpy()[out_channels]).to(device)

          layer_module.num_features = len(out_channels)
        if isinstance(layer_module, nn.Linear):
          break

def get_indices_topk(layer_bounds,layer_num,prune_limit,prune_value):

      i=layer_num
      indices=prune_value[i]

      p=len(layer_bounds)
      if (p-indices)<prune_limit:
         prune_value[i]=p-prune_limit
         indices=prune_value[i]

      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[:indices]
      return k

def get_indices_bottomk(layer_bounds,i,prune_limit):

      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[-prune_limit:]
      return k

def custom_loss(outputs, labels, model, lambda_l1=0.00001):
    lambda_l1 = 0.00001
    l1_norm = 0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param))
    ce_loss = F.cross_entropy(outputs, labels)
    total_loss = ce_loss + lambda_l1 * l1_norm
    return total_loss

def calculate_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params

def train(args, model, device, train_loader, optimizer):
    model.train()
    train_acc = 0
    total_train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        # loss = F.cross_entropy(output, target)
        loss = custom_loss(output, target, model, lambda_l1=0.00001)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum()
    total_train_loss /= len(train_loader.dataset)
    # PBG.pbTracker.addExtraScore(100. * correct / len(train_loader.dataset), 'train')
    train_acc = 100. * correct / len(train_loader.dataset)
    model.to(device)
    return total_train_loss, train_acc


def test(model, device, test_loader, optimizer, scheduler, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        # test_loss, correct, len(test_loader.dataset),
        # 100. * correct / len(test_loader.dataset)))

    # PBG.pbTracker.addTestScore(100. * correct / len(test_loader.dataset),
    # "Test_Accuracy")
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def evaluate(model, device, valid_loader, optimizer, scheduler, args):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum()

    val_loss /= len(valid_loader.dataset)
    # model, restructured, trainingComplete = PBG.pbTracker.addValidationScore(100. * correct / len(valid_loader.dataset),
    # model)
    # model.to(device)
    # if(restructured):
    #     optimArgs = {'params':model.parameters(),'lr':args.lr}
    #     schedArgs = {'step_size':1, 'gamma': args.gamma}
    #     optimizer, scheduler = PBG.pbTracker.setupOptimizer(model, optimArgs, schedArgs)
    val_accuracy = 100. * correct / len(valid_loader.dataset)
    return val_loss, val_accuracy, model, optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--save-name', type=str, default='PB')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
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

    pruningTimes = 1
    norm_mean, norm_var = 0.0, 1.0

    N = 1
    dendrites = 3
    prune_limits = [1] * 2 * 5 * dendrites
    prune_value = [1] * 2 * dendrites + [2] * 2 * dendrites + [4] * 2 * dendrites  + [8] * 2 * dendrites

    total_layers = 18 * dendrites
    total_convs = 8 * dendrites
    total_blocks = 5

    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    th.backends.cudnn.deterministic = True

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
    dataset1 = datasets.EMNIST(root='../.././data', split='balanced', train=True, download=True, transform=transform_train)
    dataset2 = datasets.EMNIST(root='../.././data',  split='balanced', train=False, download=True, transform=transform_test)

    test_size = len(dataset2)
    train_size = len(dataset1) - test_size
    val_size = test_size

    train_subset, val_subset = torch.utils.data.random_split(dataset1, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_subset, **train_kwargs)
    val_loader   = torch.utils.data.DataLoader(val_subset, **test_kwargs)
    test_loader  = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    total_step = len(train_loader)

    #Set up some global parameters for PAI code
    # PBG.switchMode = PBG.doingHistory # This is when to switch between PAI and regular learning
    #PBG.retainAllPB = True
    # PBG.nodeIndex = 1 # This is the index of the nodes within a layer
    # PBG.inputDimensions = [-1, 0, -1, -1] #this is the shape of inputs, for a standard conv net this will work
    # PBG.nEpochsToSwitch = 10  #This is how many normal epochs to wait for before switching modes.  Make sure this is higher than your schedulers patience.
    # PBG.pEpochsToSwitch = 10  #Same as above for PAI epochs
    # PBG.capAtN = True #Makes sure subsequent rounds last max as long as first round
    # PBG.initialHistoryAfterSwitches = 2
    # PBG.testSaves = True
    # PBG.testingDendriteCapacity = False

    PBG.moduleNamesToConvert.append('BasicBlock')

    #Create the model
    model = models.resnet18(num_classes == num_classes)
    print(model)
    #model = PBM.ResNetPB(model)
    # model = PBU.initializePB(model)
    model = model.to(device)
    # decision_count=th.ones((total_convs))



    #Setup the optimizer and scheduler
    # PBG.pbTracker.setOptimizer(optim.Adadelta)
    # PBG.pbTracker.setScheduler(StepLR)
    # optimArgs = {'params':model.parameters(),'lr':args.lr}
    # schedArgs = {'step_size':20, 'gamma': args.gamma}
    # optimizer, scheduler = PBG.pbTracker.setupOptimizer(model, optimArgs, schedArgs)

    optimizer = th.optim.Adadelta(params=model.parameters(),lr=args.lr)
    scheduler = StepLR(optimizer, step_size = 20, gamma= args.gamma)

    excel_path = f"results25PAIPruneInit.xlsx"

    if not os.path.exists(excel_path):
        df_init = pd.DataFrame(columns=[
            "Pruning Iteration",
            "Train Acc", "Val Acc", "Test Acc",
            "Train Loss", "Val Loss", "Test Loss",
            "Param Count, Trainable Param Count"
        ])
        df_init.to_excel(excel_path, index=False)
    a=[]
    for layer_name, layer_module in model.named_modules():
        if(isinstance(layer_module, th.nn.Conv2d) and not layer_name.startswith("conv1") and layer_name.find('conv1')!=-1):
          a.append(layer_module)

    pruningTimes = 0
    continue_pruning=True
    best_test_acc=0
    decision=True
    best_test_acc= 0.0
    continue_pruning=True
    while(continue_pruning==True):

        if(continue_pruning==True):


            with th.no_grad():

                l1norm=[]
                l_num=0
                for layer_name, layer_module in model.named_modules():

                    if(isinstance(layer_module, th.nn.Conv2d) and not layer_name.startswith('conv1') and layer_name.find('conv1')!=-1):
                        temp=[]
                        filter_weight=layer_module.weight.clone()

                        for k in range(filter_weight.size()[0]):
                          temp.append(float("{:.6f}".format((filter_weight[k,:,:,:]).norm(1).item())))

                        l1norm.append(temp)
                        l_num+=1
                        # print(layer_name)

                layer_bounds1=l1norm

            inc_indices=[]
            for i in range(len(layer_bounds1)):
                imp_indices=get_indices_bottomk(layer_bounds1[i],i,prune_limits[i])
                inc_indices.append(imp_indices)



            unimp_indices=[]
            dec_indices=[]
            for i in range(len(layer_bounds1)):
                temp=[]
                temp=get_indices_topk(layer_bounds1[i],i,prune_limits[i],prune_value)
                unimp_indices.append(temp[:])
                temp.extend(inc_indices[i])
                dec_indices.append(temp)

            print('selected  UNIMP indices ',unimp_indices)

            remaining_indices=[]
            for i in range(total_convs):
                temp=[]
                for j in range(a[i].weight.shape[0]):
                  if (j not in unimp_indices[i]):
                    temp.extend([j])
                remaining_indices.append(temp)

            with th.no_grad():

                l1norm=[]
                l_num=0
                for layer_name, layer_module in model.named_modules():

                    if(isinstance(layer_module, th.nn.Conv2d) and not layer_name.startswith('conv1') and layer_name.find('conv1')!=-1):
                        temp=[]
                        filter_weight=layer_module.weight.clone()
                        for k in range(filter_weight.size()[0]):
                                temp.append(float("{:.6f}".format((filter_weight[k,:,:,:]).norm(1).item())))
                        l1norm.append(temp)
                        l_num+=1

                layer_bounds1=l1norm

            with th.no_grad():

                if(continue_pruning==True):
                  prune_filters(model, remaining_indices)
                  print(model)

                else:
                  break

                for i in range(len(layer_bounds1)):
                    if(a[i].weight.shape[0]<= prune_limits[i]):
                      decision_count[:]=0
                      break

            if(continue_pruning==True):
                optimizer = th.optim.Adadelta(params=model.parameters(),lr=args.lr)
                scheduler = StepLR(optimizer, step_size = 20, gamma = args.gamma)

            c_epochs=0
            best_state_dict = None
            print("Training the model after pruning")
            best_val_acc = 0
            best_state_dict = None

            for c_epochs in range(1, args.epochs + 1):
                train_acc = []
                total_train_loss = 0.0

                train_loss_val, train_acc_val = train(args, model, device, train_loader, optimizer)
                test_loss_val, test_acc_val = test(model, device, test_loader, optimizer, scheduler, args)
                val_loss_val, val_acc_val, model, optimizer, scheduler = evaluate(model, device, val_loader, optimizer, scheduler, args)
                # if trainingComplete:
                #     break
                scheduler.step()

                if val_acc_val > best_val_acc:
                    best_val_acc = val_acc_val
                    best_state_dict = deepcopy(model.state_dict())

                print('\n--- Epoch: {} | Train acc: {:.2f}% | Train loss: {:.4f} | Val acc: {:.2f}% | Val loss: {:.4f} | Test acc: {:.2f} | Test loss: {:.4f}'
                    .format(c_epochs, train_acc_val, train_loss_val, val_acc_val, val_loss_val, test_acc_val, test_loss_val))

            print(f"Best Validation Accuracy for pruning iteration {pruningTimes} is: ", best_val_acc)
            model.load_state_dict(best_state_dict)
            train_loss_val, train_acc_val = train(args, model, device, train_loader, optimizer)
            test_loss_val, test_acc_val = test(model, device, test_loader, optimizer, scheduler, args)
            val_loss_val, val_acc_val, model, optimizer, scheduler = evaluate(model, device, val_loader, optimizer, scheduler, args)
            param_count = calculate_parameters(model)

            print("Final Test Loss: {:.4f} | Final Test Accuracy: {:.2f}% | Params: {:.2f}".format(test_loss_val, test_acc_val, param_count))

            new_row = {
            "Pruning Iteration": pruningTimes,
            "Train Acc": train_acc_val,
            "Val Acc": val_acc_val,
            "Test Acc": test_acc_val,
            "Train Loss": train_loss_val,
            "Val Loss": val_loss_val,
            "Test Loss": test_loss_val,
            "Param Count": param_count
            }

            df_existing = pd.read_excel(excel_path)
            df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
            df_existing.to_excel(excel_path, index=False)

            torch.save(model, f"models25PAIPruneInit/{pruningTimes}_pruned_model.pth")
            pruningTimes += 1
            # exit(1)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()