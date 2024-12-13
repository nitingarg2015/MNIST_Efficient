import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self, dropout_rate = 0):
        super(MNISTModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)          #RF: 3, nout: 28
        self.bn1 = nn.BatchNorm2d(8)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)         #RF: 5, nout: 28
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(dropout_rate)

        # self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.dropout3 = nn.Dropout(dropout_rate)

        self.maxpool1 = nn.MaxPool2d(2, 2)                              #RF: 6, nout: 14
        self.ant1 = nn.Conv2d(16,8,kernel_size=1)                       #RF: 6, nout: 14

        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, padding=1)         #RF: 10, nout: 14
        self.bn4 = nn.BatchNorm2d(16)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.conv5 = nn.Conv2d(16, 20, kernel_size=3, padding=1)        #RF: 14, nout: 14
        self.bn5 = nn.BatchNorm2d(20)
        self.dropout5 = nn.Dropout(dropout_rate)

        self.maxpool2 = nn.MaxPool2d(2, 2)                              #RF: 16, nout: 7
        self.ant2 = nn.Conv2d(20,8,kernel_size=1)                       #RF: 16, nout: 7

        self.conv6 = nn.Conv2d(8, 12, kernel_size=3, padding=0)         #RF: 24, nout: 5
        self.bn6 = nn.BatchNorm2d(12)         
        self.dropout6 = nn.Dropout(dropout_rate)
        self.conv7 = nn.Conv2d(12, 10, kernel_size=3, padding=0)        #RF: 32, nout: 3
        self.avgpool = nn.AvgPool2d(2, 2)                               #RF: 36, nout: 1

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(self.bn1(self.relu(self.conv1(x))))
        x = self.dropout2(self.bn2(self.relu(self.conv2(x))))
        # x = self.dropout3(self.bn3(self.relu(self.conv3(x))))
    
        x = self.maxpool1(self.relu(x))
        x = self.ant1(x)

        x = self.dropout4(self.bn4(self.relu(self.conv4(x))))
        x = self.dropout5(self.bn5(self.relu(self.conv5(x))))

        x = self.maxpool2(self.relu(x))
        x = self.ant2(x)

        x = self.dropout6(self.bn6(self.relu(self.conv6(x))))
        x = self.conv7(x)
        x = self.avgpool(x)
      
        x = x.view(-1, 10 * 1 * 1)    
        
        return torch.log_softmax(x, dim=1)
        # return x
    
    def to_device(self):
        return self.to('cpu')

    
if __name__ == "__main__":
    model = MNISTModel()
    print(summary(model, (1, 32, 32)))
