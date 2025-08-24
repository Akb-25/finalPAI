from __future__ import print_function
import os
import pandas as pd
import argparse
import numpy as np
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

def sort(layer_aggregated_votes, conv_layers, k):
    layer_result = {}
    for layer in conv_layers:
        agg_votes = layer_aggregated_votes[layer]
        k_indices=int((agg_votes.size) * k)
        flat_indices = np.argsort(agg_votes.ravel())[-k_indices:]
        indices = np.unravel_index(flat_indices, agg_votes.shape)
        result = np.zeros_like(agg_votes)
        result[indices] = 1
        layer_result[layer] = result

    return layer_result

def normalize_cifar10(train_data, test_data):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return transform_train, transform_test


def calculate_l1_norm_of_linear_outputs(model):
    l1_normalisation_values = {}
    for name, layer in model.named_modules():
        # print("Named modules: ", name)
        if isinstance(layer, nn.Linear):
            print("Name is: ", name)
            weights = layer.weight
            l1_norm_of_neurons = torch.sum(torch.abs(weights), dim=1).tolist()
            l1_normalisation_values[name] = l1_norm_of_neurons
    print(len(l1_normalisation_values))
    # exit(0)
    return l1_normalisation_values

def calculate_l1_norm_of_linear_inputs(model):
    l1_normalisation_values = {}
    for name, layer in model.named_modules():
        # print("Named modules: ", name)
        if isinstance(layer, nn.Linear):
            weights = layer.weight
            l1_norm_of_inputs = torch.sum(torch.abs(weights), dim=0).tolist()
            l1_normalisation_values[name] = l1_norm_of_inputs
    return l1_normalisation_values


def calculate_threshold_l1_norm(values, percentage_to_prune):
    threshold_values = {}
    for layer_name, vals in values.items():
        sorted_vals = sorted(vals)
        threshold_index = int(len(sorted_vals) * percentage_to_prune)
        threshold_value = sorted_vals[threshold_index]
        threshold_values[layer_name] = threshold_value
    return threshold_values

def print_conv_layer_shapes(model):
    print("\nLayer and shape of the filters \n -----------------------------")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"Conv layer: {name}, Weight shape: {module.weight.shape}  Bias shape: {module.bias.shape if module.bias is not None else 'No bias'}")

def calculate_l1_norm_of_filters(model):
    l1_normalisation_values={}
    for name,layer in model.named_modules():
        if isinstance(layer,nn.Conv2d):
            filters=layer.weight
            l1_norm_of_filter=[]
            for idx,filter in enumerate(filters):
                l1_norm=torch.sum(torch.abs(filter)).item()
                l1_norm_of_filter.append(l1_norm)
            l1_normalisation_values[name]=l1_norm_of_filter
    print("Conv l1 norms length is: ", len(l1_normalisation_values))
    return l1_normalisation_values

def calculate_threshold_l1_norm_of_filters(l1_normalisation_values,percentage_to_prune):
    threshold_values={}
    for filter_ in l1_normalisation_values:
        filter_values=l1_normalisation_values[filter_]
        sorted_filter_values=sorted(filter_values)
        threshold_index=int(len(filter_values)*percentage_to_prune)
        threshold_value=sorted_filter_values[threshold_index]
        threshold_values[filter_]=threshold_value
    return threshold_values

def index_remove(tensor, dim, index, removed=False):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size
    new_size = size_

    select_index = list(set(range(tensor.size(dim))) - set(index))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))

    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor


def get_new_conv(in_channels, conv, dim, channel_index, independent_prune_flag=False):

    new_conv = torch.nn.Conv2d(in_channels=in_channels,
                                   out_channels=int(conv.out_channels - len(channel_index)),
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)

    new_conv.weight.data = index_remove(conv.weight.data, 0, channel_index)
    # new_conv.bias.data = index_remove(conv.bias.data, 0, channel_index)

    return new_conv

def prune_layer(layer, outputs_to_prune, inputs_to_prune):
    in_features = layer.in_features - len(inputs_to_prune)
    out_features = layer.out_features - len(outputs_to_prune)

    new_linear_layer = nn.Linear(in_features, out_features, bias=True)

    keep_outputs = list(set(range(layer.out_features)) - set(outputs_to_prune))
    keep_inputs = list(set(range(layer.in_features)) - set(inputs_to_prune))


    new_linear_layer.weight.data = layer.weight.data[keep_outputs][:, keep_inputs]
    new_linear_layer.bias.data = layer.bias.data[keep_outputs]

    output_weights=new_linear_layer.out_features
    return new_linear_layer,output_weights

def set_nested_attr(obj, attr_path, new_value):
    parts = attr_path.split(".")
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    setattr(obj, parts[-1], new_value)

def prune_filters(model,threshold_values,l1_norm_inputs,l1_norm_outputs,threshold_inputs,threshold_outputs):
    global last_layer
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    filters_to_remove=[]
    next_channel=3
    for name,layer in model.named_modules():
        filters_to_remove=[]
        if isinstance(layer,nn.Conv2d):
            print("Name is: ", name)
            filters=layer.weight
            num_filters_to_prune=0
            print(threshold_values[name])
            for idx, filter in enumerate(filters):
                l1_norm = torch.sum(torch.abs(filter)).item()
                if l1_norm < threshold_values[name]:
                    num_filters_to_prune+=1
                    layer.weight.data[idx].zero_()
                    filters_to_remove.append(idx)

            if num_filters_to_prune == 0:
                for idx, filter in enumerate(filters):
                    l1_norm = torch.sum(torch.abs(filter)).item()
                    if l1_norm <= threshold_values[name]:
                        num_filters_to_prune+=1
                        layer.weight.data[idx].zero_()
                        filters_to_remove.append(idx)

            if num_filters_to_prune > 0:
                in_channels = next_channel
                out_channels = layer.out_channels - num_filters_to_prune
                new_conv_layer=get_new_conv(in_channels,layer,0,filters_to_remove).to(device)
                set_nested_attr(model, name, new_conv_layer)

                # setattr(model, name, new_conv_layer)
                next_channel=out_channels

        elif isinstance(layer, nn.BatchNorm2d):
            print("Name is: ", name)
            new_batch_norm_2d_layer=nn.BatchNorm2d(num_features=next_channel).to(device)
            # setattr(model,name,new_batch_norm_2d_layer)
            set_nested_attr(model, name, new_batch_norm_2d_layer)
            del new_batch_norm_2d_layer

        elif isinstance(layer, nn.BatchNorm1d):
            print("Name is: ", name)
            new_batch_norm_1d_layer=nn.BatchNorm1d(num_features=next_channel).to(device)
            # setattr(model,name,new_batch_norm_1d_layer)
            set_nested_attr(model, name, new_batch_norm_1d_layer)
            del new_batch_norm_1d_layer

        elif isinstance(layer, nn.Linear):
            print("Name is: ", name)
            print(last_layer)
            if layer==last_layer:
                outputs_to_prune=[]
            else:
                outputs_to_prune = [idx for idx, l1 in enumerate(l1_norm_outputs[name]) if l1 < threshold_outputs[name]]
            inputs_to_prune = [idx for idx, l1 in enumerate(l1_norm_inputs[name]) if l1 < threshold_inputs[name]]
            new_layer,next_channel= prune_layer(layer, outputs_to_prune, inputs_to_prune)
            print("================================================")
            print(layer)
            print(threshold_inputs[name])
            print(threshold_outputs[name])
            print(outputs_to_prune)
            print(inputs_to_prune)
            print(new_layer)
            print("================================================")
            # setattr(model, name, new_layer)
            set_nested_attr(model, name, new_layer)
    return model

def update_inputs_channels(model):
    prev_channels=3
    for name, module in model.named_modules():
        if isinstance(module,nn.Conv2d):
            in_channels=prev_channels
            module.weight.data = module.weight.data[:, :in_channels, :, :]
            module.in_channels=in_channels
            prev_channels=module.out_channels
    return model


def prune_model(model,pruning_rate,l1_norm_inputs,l1_norm_outputs,threshold_inputs,threshold_outputs):
    l1_norm_values=calculate_l1_norm_of_filters(model)

    # print("Conv values are ", l1_norm_values)
    # exit(0)
    threshold_values=calculate_threshold_l1_norm_of_filters(l1_norm_values,pruning_rate)
    model=prune_filters(model,threshold_values,l1_norm_inputs,l1_norm_outputs,threshold_inputs,threshold_outputs)
    model=update_inputs_channels(model)
    return model

def check_pruning(model):
  print("\nLayer and filter sizes \n ------------------------------------")
  for name,module in model.named_modules():
    if isinstance(module,nn.Conv2d):
      print(f"Layer: {name}, Filter Size: {module.out_channels}")


def l1_norm(model):
    '''
    l1 = 0
    for param in model.parameters():
        l1 += torch.sum(torch.abs(param))
    return l1
    '''
    l2 = 0
    for param in model.parameters():
        l2 += torch.sum(param ** 2)
    return torch.sqrt(l2)

width = float(0.3333333)

class CIFAR10VGG(nn.Module):
    def __init__(self):
        super(CIFAR10VGG, self).__init__()
        self.num_classes = 47
        self.weight_decay = 0.0005

        self.conv1 = PBG.PBSequential([nn.Conv2d(1, int(64*width), kernel_size=3, padding=4, bias=False),
        nn.BatchNorm2d(int(64*width))])
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = PBG.PBSequential([nn.Conv2d(int(64*width), int(64*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(64*width))])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = PBG.PBSequential([nn.Conv2d(int(64*width), int(128*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(128*width))])
        self.dropout2 = nn.Dropout(0.4)
        self.conv4 = PBG.PBSequential([nn.Conv2d(int(128*width), int(128*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(128*width))])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = PBG.PBSequential([nn.Conv2d(int(128*width), int(256*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(256*width))])
        self.dropout3 = nn.Dropout(0.4)
        self.conv6 = PBG.PBSequential([nn.Conv2d(int(256*width), int(256*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(256*width))])
        self.dropout4 = nn.Dropout(0.4)
        self.conv7 = PBG.PBSequential([nn.Conv2d(int(256*width), int(256*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(256*width))])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = PBG.PBSequential([nn.Conv2d(int(256*width), int(512*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(512*width))])
        self.dropout5 = nn.Dropout(0.4)
        self.conv9 = PBG.PBSequential([nn.Conv2d(int(512*width), int(512*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(512*width))])
        self.dropout6 = nn.Dropout(0.4)
        self.conv10 = PBG.PBSequential([nn.Conv2d(int(512*width), int(512*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(512*width))])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = PBG.PBSequential([nn.Conv2d(int(512*width), int(512*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(512*width))])
        self.dropout7 = nn.Dropout(0.4)
        self.conv12 = PBG.PBSequential([nn.Conv2d(int(512*width), int(512*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(512*width))])
        self.dropout8 = nn.Dropout(0.4)
        self.conv13 = PBG.PBSequential([nn.Conv2d(int(512*width), int(512*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(512*width))])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout9 = nn.Dropout(0.5)

        self.fc1 = PBG.PBSequential([nn.Linear(int(512*width), int(512*width)),
        nn.BatchNorm1d(int(512*width))])
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(int(512*width), self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu((self.conv2(x)))
        x = self.pool1(x)

        x = F.relu((self.conv3(x)))
        x = self.dropout2(x)
        x = F.relu((self.conv4(x)))
        x = self.pool2(x)

        x = F.relu((self.conv5(x)))
        x = self.dropout3(x)
        x = F.relu((self.conv6(x)))
        x = self.dropout4(x)
        x = F.relu((self.conv7(x)))
        x = self.pool3(x)
        x = F.relu((self.conv8(x)))
        x = self.dropout5(x)
        x = F.relu((self.conv9(x)))
        x = self.dropout6(x)
        x = F.relu((self.conv10(x)))
        x = self.pool4(x)

        x = F.relu((self.conv11(x)))
        x = self.dropout7(x)
        x = F.relu((self.conv12(x)))
        x = self.dropout8(x)
        x = F.relu((self.conv13(x)))
        x = self.pool5(x)
        x = self.dropout9(x)

        x = x.view(x.size(0), -1)
        x = F.relu((self.fc1(x)))
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        return x

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
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
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

    prune_limits = [1] * 2 * 5
    prune_value = [1] * 2  + [2] * 2  + [4] * 2   + [8] * 2

    total_layers = 18
    total_convs = 8
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
    
    params={
    "train_batch_size":128,
    "test_batch_size":128,
    "learning_rate":0.1,
    "num_epochs":250,
    "pruning_rate":0.05,
    "lambda_l1":10000,
    }

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    #Create the model
    model = CIFAR10VGG().to(device)
    # model = PN.loadPAIModel(model, 'ThirdTwoDendrites.pt')
    print(model)

    #model = PBM.ResNetPB(model)
    # model = PBU.initializePB(model)
    model = model.to(device)
    # decision_count=th.ones((total_convs))
    layers = list(model.modules())
    last_layer=layers[-1]


    #Setup the optimizer and scheduler
    # PBG.pbTracker.setOptimizer(optim.Adadelta)
    # PBG.pbTracker.setScheduler(StepLR)
    # optimArgs = {'params':model.parameters(),'lr':args.lr}
    # schedArgs = {'step_size':20, 'gamma': args.gamma}
    # optimizer, scheduler = PBG.pbTracker.setupOptimizer(model, optimArgs, schedArgs)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)#, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    excel_path = f"train_results_r125_nonpai_pruned_pretrain.xlsx"

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

    num_filters = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            num_filters.append(layer.out_channels)

    pruningTimes = 0
    continue_pruning=True
    best_test_acc=0
    decision=True
    best_test_acc= 0.0
    continue_pruning=True
    best_state_dict = None
    best_val_acc = 0

    l1_norm_outputs = calculate_l1_norm_of_linear_outputs(model)
    l1_norm_inputs = calculate_l1_norm_of_linear_inputs(model)
    threshold_outputs = calculate_threshold_l1_norm(l1_norm_outputs, params["pruning_rate"])
    threshold_inputs = calculate_threshold_l1_norm(l1_norm_inputs, params["pruning_rate"])

    print_conv_layer_shapes(model)

    model=prune_model(model,params["pruning_rate"],l1_norm_inputs,l1_norm_outputs,threshold_inputs,threshold_outputs)
    print(model)
    print_conv_layer_shapes(model)

    check_pruning(model)



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

    torch.save(model, f"models125NonPAIPrunePretrain/{pruningTimes}_pruned_model.pth")
    pruningTimes += 1

    while(continue_pruning==True):


            c_epochs=0
            best_state_dict = None
            print("Training the model after pruning")
            best_val_acc = 0
            best_state_dict = None

            l1_norm_outputs = calculate_l1_norm_of_linear_outputs(model)
            l1_norm_inputs = calculate_l1_norm_of_linear_inputs(model)
            threshold_outputs = calculate_threshold_l1_norm(l1_norm_outputs, params["pruning_rate"])
            threshold_inputs = calculate_threshold_l1_norm(l1_norm_inputs, params["pruning_rate"])

            print_conv_layer_shapes(model)

            model=prune_model(model,params["pruning_rate"],l1_norm_inputs,l1_norm_outputs,threshold_inputs,threshold_outputs)
            print(model)
            print_conv_layer_shapes(model)

            check_pruning(model)



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

            torch.save(model, f"models125NonPAIPrunePretrain/{pruningTimes}_pruned_model.pth")
            pruningTimes += 1

if __name__ == '__main__':
    main()