from torch import nn, optim,save
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image

class EmotionNetwork(nn.Module):
    def __init__(self): 
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(48*48, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 7), 
            nn.Softmax()
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        output = self.flatten(output)
        return output


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
training_set = torchvision.datasets.FER2013('./data', split="train", transform=transform)
validation_set = torchvision.datasets.FER2013('./data', split="test", transform=transform)

training_loader = DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=4, shuffle=False)

model = EmotionNetwork() 
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# for i, data in enumerate(training_loader):
#     inputs, expected = data #labels are a tensor of expected outputs of *batch_size* inputs
#     optimizer.zero_grad()
#     outputs = model(inputs)
#     loss = loss_fn(outputs,expected)
#     loss.backward()
#     optimizer.step()
#DIMENSION WORKS JUST CHANGE THIS THING LATER (the state.dict() )
save(model.state_dict(),"model/trained.pth")
#set model = EmotionNetwork(), then if x is the input, call model(x) to get output back

#save into a .pth, contains the modified weights and biases 

# 0: Angry
# 1: Disgust
# 2: Fear
# 3: Happy
# 4: Sad
# 5: Surprise
# 6: Neutral