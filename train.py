import math
import os, sys
import argparse
import warnings
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.nn import CrossEntropyLoss
import torchvision.transforms as transforms
from multiprocessing.reduction import ForkingPickler

from utils.dataset import Dataset
from utils.model import GNCNN
from utils.visualization import Performance_Visualization

# prevent showing warnings
warnings.filterwarnings("ignore")

# print torch and cuda information
print('=========== torch & cuda infos ================')
print('torch version : ' + torch.__version__)
print('available: ' + str(torch.cuda.is_available()))
print('count: ' + str(torch.cuda.device_count()))
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--optimizer', default='sgd', type=str)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--num_classes', default=2, type=int)
parser.add_argument('--save_dir', default='weights/', type=str)
parser.add_argument('--model_name', default='gncnn', type=str)
parser.add_argument('--num_worker', default=16, type=int)
args = parser.parse_args()

# Params
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
NUM_WORKER = args.num_worker
SAVE_DIR = args.save_dir
MODEL_DIR = '{}{}_{}.pth'.format(args.save_dir, args.model_name, args.optimizer)
NUM_CLASSES = args.num_classes
LEARNING_RATE = args.learning_rate

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# =============================================================================
# Preprocessing
transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_valid = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Load Data
train_ds = Dataset('dataset/train', transform=transform_train)
valid_ds = Dataset('dataset/test', transform=transform_valid)

trainloader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# =============================================================================
# Call out gpu is possible
torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('{} will be used in the training process !!!'.format(device))


KV = torch.tensor([ [-1,2,-2,2,-1],
                    [2,-6,8,-6,2],
                    [-2,8,-12,8,-2],
                    [2,-6,8,-6,2],
                    [-1,2,-2,2,-1]])/12.
KV = KV.view(1,1,5,5).to(device=device, dtype=torch.float)
KV = torch.autograd.Variable(KV, requires_grad=False)

def gaussian1(x):
	mean = torch.mean(x)
	std = torch.std(x)
	return torch.exp(-((x-mean)**2)/(torch.std(x))**2) 
def gaussian2(x):
    mean = torch.mean(x)
    std = torch.std(x)
    return 0.5 * torch.exp(-((x-mean)**2)/(torch.std(x))**2) 

model = GNCNN(KV, gaussian1, gaussian2)
model.to(device) 

if device == "cuda":
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# =============================================================================
# optimizer, loss function, & learning rate scheduler
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
if args.optimizer == "rmsprop":
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
elif args.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# load weight
best_acc, start_epoch = 0, 0
try:
    assert os.path.isdir(SAVE_DIR), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(MODEL_DIR)
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']
    # if start_epoch >= EPOCHS:
        #start_epoch = 0
    print('weights loaded!')
except:
    pass


# =============================================================================
# 关闭pytorch的shared memory功能 (Bus Error)
# Ref: https://github.com/huaweicloud/dls-example/issues/26
for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]


# =============================================================================
# Training and Valid function
def train(epoch):
    global train_losses, train_accs
    model.train()
    train_loss, correct, total = 0, 0, 0
    bar_len_total, batch_total = 30, len(trainloader)
    print('Epoch {}/{}'.format(epoch + 1, EPOCHS))
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # call out gpu tp process data
        inputs, targets = inputs.to(device), targets.argmax(axis=1).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # training loss and accuracy
        loss = criterion(outputs, targets)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_acc = 100.*correct/total
        
        # optimize model
        loss.backward()
        optimizer.step()
        
        # showing current progress
        bar_len = math.floor(bar_len_total * (batch_idx + 1) / batch_total)  - 1
        print('{}/{} '.format(batch_idx + 1, batch_total) + 
              '[' + '=' * bar_len + '>' + '.' * (bar_len_total - (bar_len + 1))  + '] ' + 
              '- train loss : {:.3f} '.format(train_loss / (batch_idx + 1)) + 
              '- train acc : {:.3f} '.format(train_acc)
             , end='\r')  
        
    
    # record the final training loss and acc in this epoch
    train_losses.append(train_loss / (batch_idx + 1))
    train_accs.append(train_acc)
    print('', end='')

def valid(epoch):
    global best_acc, train_losses, train_accs, valid_losses, valid_accs
    model.eval()
    valid_loss, correct, total = 0, 0, 0
    bar_len_total, batch_total = 30, len(trainloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.argmax(axis=1).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # validation loss and accuracy
    valid_acc = 100.*correct/total
    
    # print result of this epoch
    train_loss, train_acc = train_losses[-1], train_accs[-1]
    print('{}/{} '.format(batch_total, batch_total) + 
          '[' + '=' * bar_len_total  + '] ' + 
          '- train loss : {:.3f} '.format(train_loss) + 
          '- train acc : {:.3f} '.format(train_acc) + 
          '- valid loss : {:.3f} '.format(valid_loss / (batch_idx + 1)) + 
          '- valid acc : {:.3f} '.format(valid_acc))
    
    # record the final training loss and acc in this epoch
    valid_losses.append(valid_loss / (batch_idx + 1))
    valid_accs.append(valid_acc)
    
    # save checkpoint if the result achieved is the best
    if valid_acc > best_acc:
        print('Best accuracy achieved, saving model to ' + MODEL_DIR)
        state = {
            'model': model.state_dict(),
            'acc': valid_acc
        }
        torch.save(state, MODEL_DIR)
        best_acc = valid_acc


train_losses, valid_losses = [], []
train_accs, valid_accs = [], []
for epoch in range(EPOCHS):
    train(epoch)
    valid(epoch)
    scheduler.step()



# Visualization
history = dict()
history['loss'], history['acc'] = train_losses, train_accs
history['val_loss'], history['val_acc'] = valid_losses, valid_accs
Performance_Visualization(history, 'result.png')