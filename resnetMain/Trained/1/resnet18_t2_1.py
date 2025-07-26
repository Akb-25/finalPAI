import torch
import torch as th
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.prune as prune
#import pandas as pd
import numpy as np
import logging
from copy import deepcopy
import csv
from time import localtime, strftime
import argparse
from torchvision import datasets, transforms
import os
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from itertools import zip_longest
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import resnet2 as models
import os


from perforatedai2 import pb_network as PN
from perforatedai import pb_globals as PBG
from perforatedai import pb_models as PBM


random_seed=42
seed = 1787
pruningTimes = 1

class Network():

    def weight_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if self.a_type == 'relu':
                init.kaiming_normal_(m.weight.data, nonlinearity=self.a_type)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'leaky_relu':
                init.kaiming_normal_(m.weight.data, nonlinearity=self.a_type)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'tanh':
                g = init.calculate_gain(self.a_type)
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'sigmoid':
                g = init.calculate_gain(self.a_type)
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            else:
                raise
                return NotImplemented

def prune_filters(indices):
      conv_layer=0
      print("The length of indices is ", len(indices))
      for i in indices:
         print(len(i))
      for layer_name, layer_module in model.named_modules():

        if(isinstance(layer_module, th.nn.Conv2d) and not layer_name.startswith('conv1')):

          if "conv1" in layer_name:
            # Retain the same weights for specified output channels
            in_channels=[i for i in range(layer_module.weight.shape[1])]
            out_channels = indices[conv_layer]
            layer_module.weight = th.nn.Parameter(
              th.FloatTensor(layer_module.weight.data.cpu().numpy()[out_channels]).to(device)
            )

          elif 'conv2' in layer_name:
            # Retain the same weights for specified input channels
            in_channels = indices[conv_layer]
            out_channels=[i for i in range(layer_module.weight.shape[0])]
            layer_module.weight = th.nn.Parameter(
                th.FloatTensor(layer_module.weight.data.cpu().numpy()[:, in_channels]).to(device)
            )
            conv_layer += 1

            # Update in_channels and out_channels based on the retained weights
          layer_module.in_channels = len(in_channels)
          layer_module.out_channels = len(out_channels)

        if (isinstance(layer_module, th.nn.BatchNorm2d) and layer_name!='bn1' and layer_name.find('bn1')!=-1):
          out_channels=indices[conv_layer]
          # Retain the same weights, biases, and statistics for specified channels
          layer_module.weight = th.nn.Parameter(
              th.FloatTensor(layer_module.weight.data.cpu().numpy()[out_channels]).to(device)
          )
          layer_module.bias = th.nn.Parameter(
            th.FloatTensor(layer_module.bias.data.cpu().numpy()[out_channels]).to(device)
          )
          layer_module.running_mean = th.FloatTensor(layer_module.running_mean.cpu().numpy()[out_channels]).to(device)
          layer_module.running_var = th.FloatTensor(layer_module.running_var.cpu().numpy()[out_channels]).to(device)

          # Update num_features to match the retained channels
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
    

def evaluate(model, valid_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = total_loss / len(valid_loader)
    accuracy = 100 * correct / total
    return loss, accuracy

def check_pruning(model):
  print("\nLayer and filter sizes \n ------------------------------------")
  for name,module in model.named_modules():
    if isinstance(module,nn.Conv2d):
      print(f"Layer: {name}, Filter Size: {module.out_channels}")

def print_remaining_filters(model):
   print("\nThe filters are \n -----------------------------------")
   for name,module in model.named_modules():
      if isinstance(module,nn.Conv2d):
         print(f"{name} has {module.out_channels} remaining filters")

def print_conv_layer_shapes(model):
    print("\nLayer and shape of the filters \n -----------------------------")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"Conv layer: {name}, Weight shape: {module.weight.shape}  Bias shape: {module.bias.shape if module.bias is not None else 'No bias'}")

def calculate_regularization_loss(model):
    regularization_loss = 0
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            filters = layer.weight
            for filter in filters:
                l2_norm = torch.norm(filter, p=2)
                regularization_loss += l2_norm
    return regularization_loss

def custom_loss(outputs, labels, model, criterion, lambda_l1=0.0000001):
    lambda_l1 = 0.00001
    l1_norm = 0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param))
    # print("L1 NORM here for model is: ", l1_norm)
    # Cross-entropy loss
    ce_loss = criterion(outputs, labels)
    # print("The cross entropy loss here is: ", ce_loss)
    # Total loss with L1 regularization
    total_loss = ce_loss + lambda_l1 * l1_norm
    # print("The addition to the cross entropy is: ", lambda_l1*l1_norm)
    # print("The total with the regularizer is: ", total_loss)
    return total_loss


def print_loss_and_custom_loss(outputs, labels, model, criterion, lambda_l1,epoch):

    l1_norm = 0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param))
    # Cross-entropy loss
    ce_loss = criterion(outputs, labels)
    # Total loss with L1 regularization
    total_loss = ce_loss - lambda_l1 * l1_norm

    print(f"\n\nThe l1 norm as loss : {l1_norm}")

    print(f"Cross entropy loss : {ce_loss}")

    print(f"Regularisation loss : (lambda_l1*l1_norm) {lambda_l1*l1_norm}")

    print(f"Total loss : (ce_loss-lambda_l1*l1_norm) {total_loss}")

    writer.add_scalar('Loss/L1_norm', l1_norm, epoch)
    writer.add_scalar('Loss/Cross_entropy', ce_loss, epoch)
    writer.add_scalar('Loss/Regularisation', lambda_l1 * l1_norm, epoch)
    writer.add_scalar('Loss/Total', total_loss, epoch)
    return total_loss


def calculate_l1_norm(model):
    l1_norm = 0.0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param))
    return l1_norm.item()


#tr_size = 300
#te_size=300
#short=True
def main(model):    
  global pruningTimes
  lamda=0.001
  prunes=0
  continue_pruning=True
  ended_epoch=0
  best_train_acc=0
  best_test_acc=0


  decision=True
  best_test_acc= 0.0
  while(continue_pruning==True):

    if(continue_pruning==True):

      
      with th.no_grad():

        #_______________________COMPUTE L1NORM____________________________________
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
                print(layer_name)

        layer_bounds1=l1norm
        
        
        
        

  #______Selecting__filters__to__regularize_____
      print("Length of layer bounds ",len(layer_bounds1))
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
        #_______________________COMPUTE L1NORM____________________________________

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
          prune_filters(remaining_indices)
          print(model)
          
        else:
          break

        #_________________________PRUNING_EACH_CONV_LAYER__________________________
        for i in range(len(layer_bounds1)):
            if(a[i].weight.shape[0]<= prune_limits[i]):
              decision_count[:]=0
              break


        prunes+=1
      
      if(continue_pruning==False):
        lamda=0
  #______________________Custom_Regularize the model___________________________
      if(continue_pruning==True):
        # optimizer = th.optim.SGD(model.parameters(), lr=optim_lr,momentum=0.9)
        # scheduler = MultiStepLR(optimizer, milestones=milestones_array, gamma=0.1)
        optimizer = th.optim.Adadelta(params=model.parameters(),lr=args.lr)
        scheduler = StepLR(optimizer, step_size = 20, gamma = args.gamma)
  
      c_epochs=0
      best_test_acc= 0.0
      best_model = None
      print("Training the model after pruning")
      for c_epochs in range(200):

          
          train_acc=[]
          for batch_num, (inputs, targets) in enumerate(train_loader):
                    model.train()
                    optimizer.zero_grad()

                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    
                    output = model(inputs)
                    # loss = criterion(output, targets)+reg
                    loss = custom_loss(output, targets, model, criterion, lambda_l1=0.01)
                    # loss = criterion(output, targets)
                    loss.backward()
                    optimizer.step()
                    with th.no_grad():

                      y_hat = th.argmax(output, 1)
                      score = th.eq(y_hat, targets).sum()
                      train_acc.append(score.item())

          with th.no_grad():

                    epoch_train_acc= (sum(train_acc)*100)/tr_size
                    test_acc=[]
                    model.eval()
                    for batch_nums, (inputs2, targets2) in enumerate(test_loader):

                        inputs2, targets2 = inputs2.cuda(), targets2.cuda()

                        output=model(inputs2)
                        y_hat = th.argmax(output, 1)
                        score = th.eq(y_hat, targets2).sum()
                        test_acc.append(score.item())

                    epoch_test_acc=(sum(test_acc)*100)/te_size
                    if(epoch_test_acc > best_test_acc ):
                        best_test_acc=epoch_test_acc
                        best_train_acc=epoch_train_acc
                        best_model = deepcopy(model)
                    # if c_epochs%25 == 0:
                    #     torch.save(model, f"modelsTrained2_1/{pruningTimes}_pruned_model.pth")
                    print('\n---------------Epoch number: {}'.format(c_epochs),
                      '---Train accuracy: {}'.format(epoch_train_acc),
                      '----Test accuracy: {}'.format(epoch_test_acc),'--------------')
                    scheduler.step()
                    
      print("Best test acc here is: ", best_test_acc)
      model = best_model
      torch.save(model, f"modelsTrained2_1/{pruningTimes}_pruned_model.pth")
      best_test_acc = 0
      best_model = None
      ended_epoch=ended_epoch+c_epochs+1
      pruningTimes += 1


if __name__ == "__main__":
    
  norm_mean, norm_var = 0.0, 1.0

  th.manual_seed(seed)
  th.cuda.manual_seed(seed)
  th.cuda.manual_seed_all(seed)
  th.backends.cudnn.deterministic = True
  th.cuda.set_device(0)

  if th.cuda.is_available():
      th.cuda.manual_seed_all(random_seed)
      device=th.device("cuda")
  else:
      device=th.device("cpu")
  print(device)
  N = 1

  batch_size_tr = 128
  batch_size_te = 128

  

  epochs = 182
  custom_epochs=15
  new_epochs=80
  optim_lr=0.0001
  milestones_array=[100]
  lamda=0.001

  # prune_limits=[6]*5*3
  # prune_value=[1]*5+[2]*5+[4]*5

  prune_limits = [1] * 2 * 5 * 5
  prune_value = [1] * 2 * 5 + [2] * 2 * 5 + [4] * 2 * 5  + [8] * 2 * 5  

  # total_layers=32
  # total_convs=15
  # total_blocks=3

  total_layers = 18 * 5
  total_convs = 8 * 5
  total_blocks = 5

  gpu = th.cuda.is_available()

  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--save-name', type=str, default='PB')
  parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                      help='input batch size for training (default: 128)')
  parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                      help='input batch size for testing (default: 128)')
  parser.add_argument('--epochs', type=int, default=128, metavar='N',
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
  use_cuda = torch.cuda.is_available()
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
  
  PBG.moduleNamesToConvert.append('BasicBlock')

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #Dataset
  dataset1 = datasets.EMNIST(root='../.././data', split='balanced', train=True, download=True, transform=transform_train)
  dataset2 = datasets.EMNIST(root='../.././data',  split='balanced', train=False, download=True, transform=transform_test)
  train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
  test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
  total_step = len(train_loader)

  model = models.resnet18(num_classes == num_classes)
    
  model = PN.loadPAIModel(model, '25Model.pt')
  # model = torch.load("modelsTrained2/11_pruned_model.pth", weights_only = False)

  model = model.to(device)

  print(model)
  decision_count=th.ones((total_convs))

  short=False
  tr_size = 112800
  te_size=18800


  activation = 'relu'


  # optimizer = th.optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=2e-4,nesterov=True)
  # scheduler = MultiStepLR(optimizer, milestones=[91,136], gamma=0.1)
  criterion = nn.CrossEntropyLoss()
  optimizer = th.optim.Adadelta(params=model.parameters(),lr=args.lr)
  scheduler = StepLR(optimizer, step_size = 20, gamma= args.gamma)

    # PBG.pbTracker.setOptimizer(optim.Adadelta)
    # PBG.pbTracker.setScheduler(StepLR)
    # optimArgs = {'params':model.parameters(),'lr':args.lr}
    # schedArgs = {'step_size':1, 'gamma': args.gamma}
    # optimizer, scheduler = PBG.pbTracker.setupOptimizer(model, optimArgs, schedArgs)


  
  #_____________________Conv_layers_________________
  a=[]
  for layer_name, layer_module in model.named_modules():
    if(isinstance(layer_module, th.nn.Conv2d) and not layer_name.startswith("conv1") and layer_name.find('conv1')!=-1):
      a.append(layer_module)
  main(model)


