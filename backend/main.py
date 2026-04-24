from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import random

app = FastAPI(title="CNN Training View API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainingRequest(BaseModel):
    epoch: int
    batch_size: int
    learning_rate: float
    dataset: str = "mnist"

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_layer_activations(self):
        return [
            self.conv1.weight.abs().mean().item(),
            self.conv2.weight.abs().mean().item(),
            self.fc1.weight.abs().mean().item(),
            self.fc2.weight.abs().mean().item(),
        ]

model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_loader = None

def get_train_loader(batch_size):
    global train_loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

@app.on_event("startup")
async def startup_event():
    get_train_loader(32)

@app.get("/")
async def root():
    return {"message": "CNN Training View API", "status": "running"}

@app.post("/api/train")
async def train_epoch(request: TrainingRequest):
    global train_loader

    if train_loader is None or train_loader.batch_size != request.batch_size:
        get_train_loader(request.batch_size)

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    optimizer.param_groups[0]['lr'] = request.learning_rate

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx >= 10:
            break

    avg_loss = total_loss / min(10, len(train_loader))
    accuracy = correct / total

    layer_activations = model.get_layer_activations()
    layer_activations_normalized = [a / max(layer_activations) for a in layer_activations]

    return {
        "epoch": request.epoch,
        "loss": round(avg_loss, 4),
        "accuracy": round(accuracy, 4),
        "layer_activations": layer_activations_normalized
    }

@app.get("/api/model/summary")
async def model_summary():
    return {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "layers": [
            {"name": "conv1", "type": "Conv2d", "params": 320},
            {"name": "conv2", "type": "Conv2d", "params": 18496},
            {"name": "fc1", "type": "Linear", "params": 401536},
            {"name": "fc2", "type": "Linear", "params": 1290},
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
