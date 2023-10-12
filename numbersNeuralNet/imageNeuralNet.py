import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#parameters
inp_size = 784 #28x28
hidden_size=500
num_classes=10
num_trainings = 2
batch_size=100
learning_rate = 0.001

trainData = torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)

testData= torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())

trainLoad = torch.utils.data.DataLoader(dataset=trainData,batch_size=batch_size,shuffle=True)

testLoad = torch.utils.data.DataLoader(dataset=testData,batch_size=batch_size,shuffle=False)

examples = iter(testLoad)
example_data, example_targets = next(examples)
print("example images vs labels")

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0],cmap='gray')
    print(example_targets[i].item())
plt.show()

class neuralNet(nn.Module):
    def __init__(self, inp_size,hidden_size,num_classes):
        super(neuralNet,self).__init__()
        self.l1 = nn.Linear(inp_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
model = neuralNet(inp_size,hidden_size,num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

numSteps = len(trainLoad)
for training in range(num_trainings):
    for i, (images,labels) in enumerate(trainLoad):
        images= images.reshape(-1,28*28).to(device)
        labels= labels.to(device)
        outputs = model(images)
        loss= criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i+1) % 100 == 0:
            print(f'Training [{training+1}/{num_trainings}], Step [{i+1}/{numSteps}], Loss: {loss.item():.4f}')

with torch.no_grad():
    correct=0
    samples=len(testLoad.dataset)
    for images,labels in testLoad:
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device) 

        outputs = model(images)

        _,predicted = torch.max(outputs,1)
        correct+= (predicted == labels).sum().item()
    percentCorrect = correct/samples
    print(f'Accuracy of the network on the {samples} test images: {100*percentCorrect} %')