from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import uuid
import time
import numpy as np
from contextlib import asynccontextmanager

# GPU support (CUDA for NVIDIA, MPS for Apple Silicon)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Optimize CPU multi-threading
torch.set_num_threads(6)

# Multi-tenancy: store models and training state per tenant
tenant_states: Dict[str, Dict[str, Any]] = {}

# Dataset configurations
DATASET_CONFIGS = {
    "mnist": {
        "input_channels": 1,
        "image_size": 28,
        "num_classes": 10,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    },
    "cifar10": {
        "input_channels": 3,
        "image_size": 32,
        "num_classes": 10,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
}

class CNNLayer(BaseModel):
    type: str  # 'conv', 'pool', 'fc', 'flatten'
    name: str
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    kernel_size: Optional[int] = None
    padding: Optional[int] = None
    pool_size: Optional[int] = None
    stride: Optional[int] = None
    units: Optional[int] = None
    activation: Optional[str] = None

class ModelConfig(BaseModel):
    layers: List[CNNLayer]
    optimizer: str = "adam"
    learning_rate: float = 0.001

class TrainingRequest(BaseModel):
    tenant_id: str
    batch_size: int = 32
    learning_rate: float = 0.001
    dataset: str = "mnist"
    epochs: int = 1

class InferenceRequest(BaseModel):
    tenant_id: str
    image_data: List[float]  # 784 for MNIST (28x28) or 3072 for CIFAR (32x32x3)

class CreateModelRequest(BaseModel):
    tenant_id: str
    config: ModelConfig
    dataset: str = "mnist"

# Global train loaders cache
train_loaders: Dict[str, torch.utils.data.DataLoader] = {}
val_loaders: Dict[str, torch.utils.data.DataLoader] = {}

def get_train_loader(dataset_name: str, batch_size: int):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cache_key = f"{dataset_name}_{batch_size}"
    if cache_key in train_loaders:
        return train_loaders[cache_key]

    config = DATASET_CONFIGS[dataset_name]

    if dataset_name == "mnist":
        dataset = datasets.MNIST('./data', train=True, download=True, transform=config["transform"])
    elif dataset_name == "cifar10":
        dataset = datasets.CIFAR10('./data', train=True, download=True, transform=config["transform"])

    # Optimized DataLoader with multiprocessing
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        persistent_workers=True,  # Keep workers alive
        pin_memory=True if device.type != "cpu" else False  # Faster GPU transfer
    )
    train_loaders[cache_key] = loader
    return loader

def get_val_loader(dataset_name: str, batch_size: int = 10):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cache_key = f"{dataset_name}_val_{batch_size}"
    if cache_key in val_loaders:
        return val_loaders[cache_key]

    config = DATASET_CONFIGS[dataset_name]

    if dataset_name == "mnist":
        dataset = datasets.MNIST('./data', train=False, download=True, transform=config["transform"])
    elif dataset_name == "cifar10":
        dataset = datasets.CIFAR10('./data', train=False, download=True, transform=config["transform"])

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True if device.type != "cpu" else False
    )
    val_loaders[cache_key] = loader
    return loader

class ConfigurableCNN(nn.Module):
    def __init__(self, config: ModelConfig, dataset: str = "mnist"):
        super().__init__()
        dataset_config = DATASET_CONFIGS.get(dataset, DATASET_CONFIGS["mnist"])
        self.layers_list = nn.ModuleList()
        self.activation_funcs = []  # List of activation functions in order
        self.flatten = None
        self.has_flatten = False

        prev_channels = dataset_config["input_channels"]
        prev_size = dataset_config["image_size"]

        for layer in config.layers:
            if layer.type == 'conv':
                self.layers_list.append(nn.Conv2d(
                    prev_channels,
                    layer.out_channels,
                    layer.kernel_size,
                    padding=layer.padding or 1
                ))
                if layer.activation == 'relu':
                    self.activation_funcs.append(nn.ReLU())
                else:
                    self.activation_funcs.append(None)
                prev_channels = layer.out_channels

            elif layer.type == 'pool':
                self.layers_list.append(nn.MaxPool2d(layer.pool_size, layer.stride or 2))
                self.activation_funcs.append(None)  # No activation after pool
                prev_size = prev_size // layer.pool_size

            elif layer.type == 'flatten':
                self.flatten = nn.Flatten()
                self.has_flatten = True
                self.activation_funcs.append(None)  # No activation after flatten itself
                prev_channels = prev_channels * prev_size * prev_size

            elif layer.type == 'fc':
                self.layers_list.append(nn.Linear(prev_channels, layer.units))
                # Note: Don't apply softmax for CrossEntropyLoss, it expects raw logits
                if layer.activation == 'relu':
                    self.activation_funcs.append(nn.ReLU())
                else:
                    self.activation_funcs.append(None)  # No activation for output layer
                prev_channels = layer.units

    def forward(self, x):
        flatten_applied = False

        for i, layer in enumerate(self.layers_list):
            # Apply flatten before FC layer if not yet done
            if isinstance(layer, nn.Linear) and self.has_flatten and not flatten_applied:
                x = self.flatten(x)
                flatten_applied = True

            # Forward through layer
            x = layer(x)

            # Apply activation if exists (ReLU only, no softmax for CrossEntropyLoss)
            activation = self.activation_funcs[i]
            if activation is not None and isinstance(activation, nn.ReLU):
                x = torch.relu(x)

        # Final flatten if no FC layers but flatten was defined
        if self.has_flatten and not flatten_applied:
            x = self.flatten(x)

        return x

def create_model_for_tenant(tenant_id: str, config: ModelConfig, dataset: str):
    model = ConfigurableCNN(config, dataset).to(device)

    # Support Adam, AdamW, and SGD optimizers
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    else:  # Default to Adam
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Learning rate scheduler: CosineAnnealing for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss()

    tenant_states[tenant_id] = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
        "config": config,
        "dataset": dataset,
        "training_history": [],
        "current_epoch": 0,
        "created_at": time.time()
    }
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create default model
    default_config = ModelConfig(
        layers=[
            CNNLayer(type="conv", name="conv1", in_channels=1, out_channels=32, kernel_size=3, padding=1),
            CNNLayer(type="pool", name="pool1", pool_size=2, stride=2),
            CNNLayer(type="conv", name="conv2", in_channels=32, out_channels=64, kernel_size=3, padding=1),
            CNNLayer(type="pool", name="pool2", pool_size=2, stride=2),
            CNNLayer(type="flatten", name="flatten"),
            CNNLayer(type="fc", name="fc1", units=128, activation="relu"),
            CNNLayer(type="fc", name="fc2", units=10, activation="softmax"),
        ]
    )
    # Preload MNIST dataset
    get_train_loader("mnist", 32)
    print(f"Preloaded MNIST dataset")
    yield
    # Cleanup on shutdown
    print("Shutting down...")

app = FastAPI(title="CNN Training View API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": str(device)
    }
    return {
        "message": "CNN Training View API",
        "status": "running",
        "gpu_info": gpu_info,
        "tenants": len(tenant_states)
    }

@app.post("/api/model/create")
async def create_model(request: CreateModelRequest):
    try:
        model = create_model_for_tenant(request.tenant_id, request.config, request.dataset)
        return {
            "tenant_id": request.tenant_id,
            "status": "created",
            "message": f"Model created for tenant {request.tenant_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
async def train(request: TrainingRequest):
    tenant_id = request.tenant_id

    # Create model if not exists for this tenant
    if tenant_id not in tenant_states:
        default_config = ModelConfig(
            layers=[
                CNNLayer(type="conv", name="conv1", in_channels=1, out_channels=32, kernel_size=3, padding=1),
                CNNLayer(type="pool", name="pool1", pool_size=2, stride=2),
                CNNLayer(type="conv", name="conv2", in_channels=32, out_channels=64, kernel_size=3, padding=1),
                CNNLayer(type="pool", name="pool2", pool_size=2, stride=2),
                CNNLayer(type="flatten", name="flatten"),
                CNNLayer(type="fc", name="fc1", units=128, activation="relu"),
                CNNLayer(type="fc", name="fc2", units=10, activation="softmax"),
            ]
        )
        create_model_for_tenant(tenant_id, default_config, request.dataset)

    state = tenant_states[tenant_id]
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    criterion = state["criterion"]

    model.train()
    train_loader = get_train_loader(request.dataset, request.batch_size)

    total_loss = 0
    correct = 0
    total = 0
    batches_processed = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        batches_processed += 1

    # Step the scheduler after each epoch
    scheduler.step()

    state["current_epoch"] += 1
    avg_loss = total_loss / batches_processed
    accuracy = correct / total

    state["training_history"].append({
        "epoch": state["current_epoch"],
        "loss": round(avg_loss, 4),
        "accuracy": round(accuracy, 4)
    })

    # Get activations from conv layers only
    layer_activations = []
    for layer in model.layers_list:
        if isinstance(layer, nn.Conv2d):
            layer_activations.append(layer.weight.abs().mean().item())

    return {
        "tenant_id": tenant_id,
        "epoch": state["current_epoch"],
        "loss": round(avg_loss, 4),
        "accuracy": round(accuracy, 4),
        "layer_activations": layer_activations
    }

@app.post("/api/inference")
async def inference(request: InferenceRequest):
    tenant_id = request.tenant_id

    if tenant_id not in tenant_states:
        raise HTTPException(status_code=404, detail=f"No model found for tenant {tenant_id}")

    state = tenant_states[tenant_id]
    model = state["model"]
    dataset = state["dataset"]

    model.eval()

    try:
        # Convert flat list to 2D array based on dataset
        config = DATASET_CONFIGS[dataset]
        image_size = config["image_size"]
        channels = config["input_channels"]

        # Reshape image_data to tensor
        if len(request.image_data) == image_size * image_size:
            # MNIST: single channel
            img_array = torch.tensor(request.image_data).reshape(1, 1, image_size, image_size)
        elif len(request.image_data) == image_size * image_size * channels:
            # CIFAR: multiple channels
            img_array = torch.tensor(request.image_data).reshape(channels, image_size, image_size)
        else:
            raise HTTPException(status_code=400, detail="Invalid image dimensions")

        # Normalize with same parameters as training
        if dataset == "mnist":
            # MNIST normalization: (x - mean) / std
            img_tensor = img_array.float().to(device)
            img_tensor = (img_tensor - 0.1307) / 0.3081
        elif dataset == "cifar10":
            # CIFAR-10 normalization: (x - 0.5) / 0.5 for each channel
            img_tensor = img_array.float().to(device)
            img_tensor = (img_tensor - 0.5) / 0.5
        else:
            img_tensor = img_array.float().to(device) / 255.0

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()

        predicted_class = int(probabilities.argmax())
        confidence = float(probabilities[predicted_class])

        return {
            "tenant_id": tenant_id,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": probabilities.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/summary/{tenant_id}")
async def model_summary(tenant_id: str):
    if tenant_id not in tenant_states:
        raise HTTPException(status_code=404, detail=f"No model found for tenant {tenant_id}")

    state = tenant_states[tenant_id]
    model = state["model"]

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "tenant_id": tenant_id,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "dataset": state["dataset"],
        "training_history": state["training_history"]
    }

@app.get("/api/datasets")
async def list_datasets():
    return {
        "datasets": [
            {"name": "mnist", "description": "手写数字识别 (28x28 grayscale)", "classes": 10},
            {"name": "cifar10", "description": "物体识别 (32x32 RGB)", "classes": 10}
        ]
    }

@app.get("/api/datasets/{dataset_name}/samples")
async def get_dataset_samples(dataset_name: str, count: int = 10):
    if dataset_name not in DATASET_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    val_loader = get_val_loader(dataset_name, count)
    samples = []
    labels = []

    for i, (data, target) in enumerate(val_loader):
        if i >= 1:  # Just get first batch
            break
        # Convert to numpy and normalize to [0, 255]
        if dataset_name == "mnist":
            # data shape: [batch, 1, 28, 28]
            img_array = (data.numpy() * 0.3081 + 0.1307).clip(0, 1) * 255
            img_array = img_array.astype(np.uint8)
            for j in range(min(count, data.shape[0])):
                samples.append(img_array[j].flatten().tolist())
                labels.append(int(target[j]))
        elif dataset_name == "cifar10":
            # data shape: [batch, 3, 32, 32], values in [-1, 1] range
            img_array = ((data.numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            # Convert from CHW to HWC
            img_array = img_array.transpose(0, 2, 3, 1)
            for j in range(min(count, data.shape[0])):
                samples.append(img_array[j].flatten().tolist())
                labels.append(int(target[j]))

    return {
        "dataset": dataset_name,
        "samples": samples,
        "labels": labels,
        "image_size": DATASET_CONFIGS[dataset_name]["image_size"],
        "channels": DATASET_CONFIGS[dataset_name]["input_channels"]
    }

@app.delete("/api/model/{tenant_id}")
async def delete_model(tenant_id: str):
    if tenant_id in tenant_states:
        del tenant_states[tenant_id]
        return {"status": "deleted", "tenant_id": tenant_id}
    raise HTTPException(status_code=404, detail=f"No model found for tenant {tenant_id}")

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "active_tenants": len(tenant_states)
    }

if __name__ == "__main__":
    import uvicorn
    # Run with GPU support if available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPUs")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)