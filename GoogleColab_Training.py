
import torch
import torchvision
import torchvision.transforms as transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assume that we are on a CUDA machine, then this should print a CUDA device:

print(device)
#--------------------
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 128, 5, 1, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv5 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv6 = nn.Conv2d(128, 256, 3, 1,1)
        self.conv7 = nn.Conv2d(640, 512, 5, 2, 1)
        self.conv8 = nn.Conv2d(512, 1024, 3, 2, 1)  
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn. Linear(1024*1*1, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x1, x2 = torch.split(x, 32, dim=1)
        x1 = F.relu(self.conv2(x1))
        x2 = F.relu(self.conv3(x2))
        x1 = F.relu(self.conv4(x1))
        x2 = F.relu(self.conv5(x2))
        x = F.relu(torch.cat((x1, x2), dim = 1)) #Nối y, z lại või dimeson =1
        x2 = F.relu(self.conv6(x2))
        y = torch.sigmoid(x1 * x2)
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x)) 
        x = self.avgpool(x)
        x = x.view(-1, 1024*1*1)
        x = self.fc1(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net()
net.to(device)
from torchsummary import summary
summary(net, (3,32,32))


#-----------------
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)


#----------------------
for epoch in range(8):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

#----------------------------
dataiter = iter(testloader)
images, labels = next(dataiter)
inputs, labels = inputs.to(device), labels.to(device)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

#------------------------------
# Kiểm tra thiết bị của input (CPU hoặc CUDA)
print(images.device)  # Kiểm tra thiết bị của input (CPU hoặc CUDA)
print(next(net.parameters()).device)  # Kiểm tra thiết bị của model

# Chuyển images và labels sang GPU
images = images.to('cuda')
labels = labels.to('cuda')  # Chuyển labels sang cùng thiết bị

# Tiếp tục tính toán
outputs = net(images)


#---------------------------
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device) 

        outputs = net(images)  # Gọi net với một đầu vào
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

#----------------------
# Đảm bảo mô hình được chuyển sang GPU
net.to('cuda')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)  # Bây giờ sẽ không còn lỗi ở dòng này
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

        # Sử dụng kích thước batch động thay vì giả định là 4
        for i in range(images.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# In kết quả
for i in range(10):
    if class_total[i] > 0:  # Kiểm tra tránh chia cho 0
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    else:
        print('Accuracy of %5s : N/A' % (classes[i]))  # Nếu không có mẫu cho lớp này
#----------------------
#-------------------
class Net(nn.Module):
	def_init__(self):
		super (Net, self)._init()
		self.conv1 = nn.Conv2d(3, 64, 5, 1, 1) # Đầu vào: 3 - Đầu ra: 64 Kennel Size: 5 Stride: 1 Padding: 1
		self.conv3 nn.Conv2d(3, 32, 3) # Đầu vào: 3 - Đầu ra: 32 Kennel Size: 3
		self.conv4 = nn.Conv2d(96, 128, 5) # Đầu vào: 96 - Đầu ra: 128 Kennel Size: 5 Stride: 1
		self.conv5 nn.Conv2d(128, 256, 3, 2) # Đầu vào: 128 Đầu ra: 256 Kennel Size: 3 Stride: 2
		self.conv6 = nn.Conv2d(256, 512, 3, 2) # Đầu vào: 256 Đầu ra: 512 Kennel Size: 3 Stride: 2
		#self.pool = nn. AvgPool2d(5, 1) # Kennel Size: 5 Stride: 1
		self.avgpool = nn.AdaptiveAvgPool2d(1) #output_size = 1
		# Dùng 1 trong 2 lệnh trên, rcm dùng lệnh dưới
		self.fc1 = nn. Linear (512*1*1, 10) # Chuyển từ 512*1*1 phần tử thành 10

		#Công thức Con = ((W - K + 2P)/S) +1 ------ W: width, k: kennel, p: padding, s: stride
		#Công thức Relu:
		#Công thức Maxpool, WH = ((H_in + 2*padding(=0) - dilation(=1) * (kennel -1) -1) / stride) +1
		#Công thức AvgPool = ((H_in + 2*padding(=0) - kennel)/ Stride) +1

	def forward(self, x):
		y = F.relu(self.conv1(x)) # [-1, 64, 30, 30] = y
		z = F.relu(self.conv3(x)) # [-1, 32, 30, 30] = z
		x = torch.cat((y, z), dim = 1) #Nối y, z lại või dimeson =1
		X = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		x = F.relu(self.conv6(x))
		x = self.avgpool(x)
		x = x.view(-1, 512 * 1 * 1)
		x = F.relu(self.fc1(x))
		return x
#-------		
net = Net()
from torchsummary import summary
summary(net, (3,32,32))