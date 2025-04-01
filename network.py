from torch import nn, optim,save,no_grad
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

class EmotionNetwork(nn.Module):
    def __init__(self): 
        super().__init__()
        self.flatten = nn.Flatten()
        self.network= nn.Sequential(
            nn.Linear(48*48, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 7),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.network(x)
        output = self.flatten(output)
        return output
def train(model,optimizer,loss_fn,training_data):
    model.train(True)
    currloss = 0
    last_loss = 0
    for j, data in enumerate(training_data):
        inputs, expected = data #labels are a tensor of expected outputs of *batch_size* inputs
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs,expected)
        loss.backward()
        optimizer.step()
        currloss += loss.item()
        if j % 1000 == 999:
            last_loss = running_loss / 1000  
            print('  batch {} loss: {}'.format(j + 1, last_loss))
            running_loss = 0
    return last_loss

def evaluate(model,validation_loader):
    currLoss = 0
    model.eval()

    with no_grad():
        for i, data in enumerate(validation_loader):
            inputs, expected = data
            outputs = model(inputs)
            loss = loss_fn(outputs, expected)
            currLoss+= loss

    avgLoss = currLoss / (i + 1)
    return avgLoss
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
training_set = torchvision.datasets.FER2013('./data', split="train", transform=transform)
validation_set = torchvision.datasets.FER2013('./data', split="test", transform=transform)

training_loader = DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=4, shuffle=False)

model = EmotionNetwork() 
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

if(__name__ == "__main__"):
    best = float("inf")
    epochs = 10
    for i in range(epochs):
        avg_loss = train(model,optimizer,loss_fn,training_loader)
        print("Epoch " + str(i) + " finished")
        avg_vloss = evaluate(model,validation_loader)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        if avg_loss < best:
            
            best = avg_loss
            save(model.state_dict(),"model/trained.pth")
    print("Training finished")

