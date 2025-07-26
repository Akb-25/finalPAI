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
import resnet as models
import pandas as pd


from perforatedai2 import pb_network as PN
from perforatedai import pb_globals as PBG
from perforatedai import pb_models as PBM


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

def custom_loss(outputs, labels, model, criterion, lambda_l1=0.0000001):
    lambda_l1 = 0.00001
    l1_norm = 0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param))
    ce_loss = criterion(outputs, labels)
    total_loss = ce_loss + lambda_l1 * l1_norm
    return total_loss

def calculate_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params

def calculate_trainable_parameters(model):
    trainable_params = 0
    for param in model.parameters():
        if param.requires_grad:
          trainable_params += param.numel()
    return trainable_params


def train(model):
    global pruningTimes
    ended_epoch=0
    best_test_acc= 0.0
    optimizer = th.optim.SGD(model.parameters(), lr=optim_lr,momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=milestones_array, gamma=0.1)
    c_epochs=0
    best_val_acc = 0
    best_train_acc = 0
    best_state_dict = None

    for c_epochs in range(200):
      model.train()
      train_acc = []
      total_train_loss = 0.0

      for batch_num, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = inputs.cuda(), targets.cuda()
        output = model(inputs)
        loss = custom_loss(output, targets, model, criterion, lambda_l1=0.00001)
        loss.backward()
        optimizer.step()
        # accumulate train loss
        total_train_loss += loss.item()

        with torch.no_grad():
          y_hat = torch.argmax(output, 1)
          score = torch.eq(y_hat, targets).sum()
          train_acc.append(score.item())
          
      with torch.no_grad():
        epoch_train_acc = (sum(train_acc) * 100) / tr_size
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        epoch_train_loss = total_train_loss / len(train_loader)  
      
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = epoch_train_acc
            best_state_dict = deepcopy(model.state_dict())
        print('\n--- Epoch: {} | Train acc: {:.2f}% | Train loss: {:.4f} | Val acc: {:.2f}% | Val loss: {:.4f}'
            .format(c_epochs, epoch_train_acc, epoch_train_loss, val_acc, val_loss))
          
        scheduler.step()
    print(f"Best Validation Accuracy for pruning iteration {pruningTimes} is: ", best_val_acc)
    model.load_state_dict(best_state_dict)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print("Final Test Loss: {:.4f} | Final Test Accuracy: {:.2f}%".format(test_loss, test_acc))
    train_loss, train_acc = evaluate(model, train_loader, criterion)
    val_loss, val_acc     = evaluate(model, val_loader, criterion)
    param_count = calculate_parameters(model)
    trainable_param_count = calculate_trainable_parameters(model)
    new_row = {
        "Pruning Iteration": pruningTimes,
        "Train Acc": train_acc,
        "Val Acc": val_acc,
        "Test Acc": test_acc,
        "Train Loss": train_loss,
        "Val Loss": val_loss,
        "Test Loss": test_loss,
        "Param Count": param_count,
        "Trainable Param Count": trainable_param_count
    }
    df_existing = pd.read_excel(excel_path)
    df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
    df_existing.to_excel(excel_path, index=False)
    torch.save(model, f"modelsTrained2_1/{pruningTimes}_pruned_model.pth")
    ended_epoch += c_epochs + 1
    pruningTimes += 1

    return model

def main(model):    
  global pruningTimes
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
          prune_filters(remaining_indices)
          print(model)
          
        else:
          break

        for i in range(len(layer_bounds1)):
            if(a[i].weight.shape[0]<= prune_limits[i]):
              decision_count[:]=0
              break


        prunes+=1
      
      if(continue_pruning==False):
        lamda=0
      if(continue_pruning==True):
        optimizer = th.optim.Adadelta(params=model.parameters(),lr=args.lr)
        scheduler = StepLR(optimizer, step_size = 20, gamma = args.gamma)
  
      c_epochs=0
      best_state_dict = None
      print("Training the model after pruning")
      best_val_acc = 0
      best_train_acc = 0
      best_state_dict = None

      for c_epochs in range(200):
        model.train()
        train_acc = []
        total_train_loss = 0.0

        for batch_num, (inputs, targets) in enumerate(train_loader):
          optimizer.zero_grad()
          inputs, targets = inputs.cuda(), targets.cuda()
          output = model(inputs)
          loss = custom_loss(output, targets, model, criterion, lambda_l1=0.00001)
          loss.backward()
          optimizer.step()
          # accumulate train loss
          total_train_loss += loss.item()

          with torch.no_grad():
            y_hat = torch.argmax(output, 1)
            score = torch.eq(y_hat, targets).sum()
            train_acc.append(score.item())

        with torch.no_grad():
          epoch_train_acc = (sum(train_acc) * 100) / tr_size
          val_loss, val_acc = evaluate(model, val_loader, criterion)
          epoch_train_loss = total_train_loss / len(train_loader)  
        
          if val_acc > best_val_acc:
              best_val_acc = val_acc
              best_train_acc = epoch_train_acc
              best_state_dict = deepcopy(model.state_dict())

          print('\n--- Epoch: {} | Train acc: {:.2f}% | Train loss: {:.4f} | Val acc: {:.2f}% | Val loss: {:.4f}'
              .format(c_epochs, epoch_train_acc, epoch_train_loss, val_acc, val_loss))
            
          scheduler.step()

      print(f"Best Validation Accuracy for pruning iteration {pruningTimes} is: ", best_val_acc)

      model.load_state_dict(best_state_dict)

      test_loss, test_acc = evaluate(model, test_loader, criterion)
      print("Final Test Loss: {:.4f} | Final Test Accuracy: {:.2f}%".format(test_loss, test_acc))

      train_loss, train_acc = evaluate(model, train_loader, criterion)
      val_loss, val_acc     = evaluate(model, val_loader, criterion)
      param_count = calculate_parameters(model)
      trainable_param_count = calculate_trainable_parameters(model)
      new_row = {
          "Pruning Iteration": pruningTimes,
          "Train Acc": train_acc,
          "Val Acc": val_acc,
          "Test Acc": test_acc,
          "Train Loss": train_loss,
          "Val Loss": val_loss,
          "Test Loss": test_loss,
          "Param Count": param_count,
          "Trainable Param Count": trainable_param_count
      }

      df_existing = pd.read_excel(excel_path)
      df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
      df_existing.to_excel(excel_path, index=False)

      torch.save(model, f"modelsTrainedBase2/{pruningTimes}_pruned_model.pth")

      ended_epoch += c_epochs + 1
      pruningTimes += 1


if __name__ == "__main__":

  random_seed=42
  seed = 1787
  pruningTimes = 1

    
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

 
  prune_limits = [1] * 2 * 5 * 5
  prune_value = [1] * 2 * 5 + [2] * 2 * 5 + [4] * 2 * 5  + [8] * 2 * 5  

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
  # iteration = 1

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

  dataset1 = datasets.EMNIST(root='.././data', split='balanced', train=True, download=True, transform=transform_train)
  dataset2 = datasets.EMNIST(root='.././data',  split='balanced', train=False, download=True, transform=transform_test)
  
  test_size = len(dataset2)
  train_size = len(dataset1) - test_size
  val_size = test_size

  train_subset, val_subset = torch.utils.data.random_split(dataset1, [train_size, val_size])

  train_loader = torch.utils.data.DataLoader(train_subset, **train_kwargs)
  val_loader   = torch.utils.data.DataLoader(val_subset, **test_kwargs)
  test_loader  = torch.utils.data.DataLoader(dataset2, **test_kwargs)

  total_step = len(train_loader)

  model = models.resnet18(num_classes == num_classes)
  # model = PN.loadPAIModel(model, '25Model.pt')
#   model = torch.load(f"modelsTrained1_{iteration}/{iteration}_pruned_model.pth", weights_only=False)
    
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

  excel_path = f"training_results2_1trained.xlsx"

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

  model = train(model)

  main(model)