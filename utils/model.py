import torch
import torch.nn as nn
import torch.nn.functional as F

class GNCNN(nn.Module):
  def __init__(self, KV, gaussian1, gaussian2):
    super(GNCNN, self).__init__()
    self.KV = KV
    self.gaussian1 = gaussian1
    self.gaussian2 = gaussian2
    self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0, bias=True)
    self.avg_pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0, bias=True)
    self.avg_pool2 =nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0, bias=True)
    self.avg_pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)

    self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0, bias=True)
    self.avg_pool4 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)

    self.conv5 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0, bias=True)
    self.avg_pool5 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)

    self.fc1 = nn.Linear(16*4*4, 128)
    self.fc2 = nn.Linear(128, 128)
    self.fc3 = nn.Linear(128, 2)

  def forward(self, x):
    prep = F.conv2d(x, self.KV, padding=2) # preprocessing
    out = self.avg_pool1(self.gaussian1(self.conv1(prep)))
    out = self.avg_pool2(self.gaussian2(self.conv2(out)))
    out = self.avg_pool3(self.gaussian2(self.conv3(out)))
    out = self.avg_pool4(self.gaussian2(self.conv4(out)))
    out = self.avg_pool5(self.gaussian2(self.conv5(out)))
    out = out.view(out.size(0), -1)

    out = F.relu(self.fc1(out))
    out = F.relu(self.fc2(out))
    out = self.fc3(out)
    return out

  def reset_parameters(self):
    for mod in self.modules():
      if isinstance(mod, nn.Conv2d):
        nn.init.xavier_uniform_(self.conv1.weight)
      elif isinstance(mod, nn.Linear):
        nn.init.kaiming_normal_(mod.weight.data)