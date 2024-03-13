import torch
import torch.nn.functional as F
from torch import nn


class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):#([128, 2048, 3])
        x = x.transpose(1, 2)#([128, 3, 2048])  batch_Size,Point_nums,position
        x = F.relu(self.bn1(self.conv1(x)))#([128, 128, 2048])
        x = F.relu(self.bn2(self.conv2(x)))#([128, 128, 2048])
        x = F.relu(self.bn3(self.conv3(x)))#([128, 256, 2048])
        x = self.bn4(self.conv4(x))#([128, 512, 2048])
        x = torch.max(x, 2, keepdim=True)[0]#(128,512,1)  [0] is only max value, [1] is only a max index  Just find the maximum value of the column dim=k, the code is to find the most useful (large) point
        x = x.view(-1, 512)#(128,512)From the original 3 attributes to the current 512 attributes
        #The above code is to select the features corresponding to the excellent points
        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))#([128, 512])--->([128, 256])  feature is change
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))#([128, 256])---->([128, 128])
        m = self.fc3_m(m)#([128, 128])---->([128, 256])
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))#([128, 512])--->([128, 256])
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))#([128, 256])---->([128, 128])
        v = self.fc3_v(v)#([128, 128])---->([128, 256])
        #Linearly transform all the features in the point cloud to achieve feature extraction
        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v#([128, 256]),([128, 256])

