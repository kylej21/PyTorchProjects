import torch

#Linear regression for f(x) = 4x+3

X= torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32)
Y=torch.tensor([7,11,15,19,23,27,31,35,39,43], dtype= torch.float32)

w= torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

def forward(x):
    return (w*x)+3
def loss(y,y_exp):
    return ((y_exp-y)**2).mean()

testVal = 100
print(f'Prediction before training: f({testVal}) = {forward(testVal).item():.3f}')

learningRate = 0.01
numTrainings=25
for training in range(numTrainings):
    y_exp=forward(X)
    error = loss(Y,y_exp)
    error.backward()
    with torch.no_grad():
        w -= learningRate* w.grad
    w.grad.zero_()
    print(f'training {training+1}: W = {w.item():.3f}, loss = {error.item():.3f}')
print(f'Prediction after all training of f({testVal}) = {forward(testVal).item():.3f}')