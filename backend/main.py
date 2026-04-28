from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import asyncio
import re
import uuid
import time
import numpy as np
from contextlib import asynccontextmanager

# Tenant ID validation pattern (alphanumeric, hyphen, underscore)
TENANT_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{8,64}$')

def validate_tenant_id(tenant_id: str) -> bool:
    """Validate tenant_id format to prevent injection."""
    if not tenant_id or len(tenant_id) > 64:
        return False
    return bool(TENANT_ID_PATTERN.match(tenant_id))

# GPU support (CUDA for NVIDIA, MPS for Apple Silicon)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("No GPU detected, using CPU")
print(f"Active device: {device}")

# Optimize CPU multi-threading
torch.set_num_threads(min(6, __import__('os').cpu_count() or 4))

# Multi-tenancy: store models and training state per tenant
tenant_states: Dict[str, Dict[str, Any]] = {}
tenant_locks: Dict[str, asyncio.Lock] = {}

MAX_TENANTS = 200

def get_tenant_lock(tenant_id: str) -> asyncio.Lock:
    if tenant_id not in tenant_locks:
        tenant_locks[tenant_id] = asyncio.Lock()
    return tenant_locks[tenant_id]

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
    "fashion_mnist": {
        "input_channels": 1,
        "image_size": 28,
        "num_classes": 10,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    },
    "cifar10": {
        "input_channels": 3,
        "image_size": 32,
        "num_classes": 10,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    },
    "cifar100": {
        "input_channels": 3,
        "image_size": 32,
        "num_classes": 100,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    },

}

DATASET_DESCRIPTIONS = {
    "mnist": {
        "name": "MNIST",
        "full_name": "Modified National Institute of Standards and Technology",
        "description": "MNIST 是最经典的深度学习入门数据集，包含 70,000 张手写数字灰度图像（60,000 训练 + 10,000 测试）。图像尺寸为 28×28 像素，涵盖数字 0-9 共 10 个类别。",
        "difficulty": "入门级",
        "paper": "LeCun et al., 1998",
        "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "channels": "灰度 (1通道)",
        "recommended_model": "1-2 个卷积层即可达到 99%+ 准确率"
    },
    "fashion_mnist": {
        "name": "Fashion-MNIST",
        "full_name": "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms",
        "description": "Fashion-MNIST 是 MNIST 的直接替代品，包含 70,000 张时尚商品的灰度图像（60,000 训练 + 10,000 测试）。图像尺寸 28×28，共 10 个类别，包括 T恤、裤子、套衫等。比 MNIST 更具挑战性，因为衣物纹理更复杂。",
        "difficulty": "初级",
        "paper": "Xiao et al., 2017 (arXiv:1708.07747)",
        "classes": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
        "channels": "灰度 (1通道)",
        "recommended_model": "2-3 个卷积层，准确率可达 90-93%"
    },
    "cifar10": {
        "name": "CIFAR-10",
        "full_name": "Canadian Institute For Advanced Research - 10 Classes",
        "description": "CIFAR-10 是经典的彩色图像分类数据集，包含 60,000 张 32×32 的 RGB 彩色图像（50,000 训练 + 10,000 测试）。涵盖飞机、汽车、鸟类、猫等 10 个类别。由于是彩色图像且分辨率较低，需要更多卷积层来提取有效特征。",
        "difficulty": "中级",
        "paper": "Krizhevsky, 2009",
        "classes": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
        "channels": "RGB (3通道)",
        "recommended_model": "3-4 个卷积层 + 数据增强，准确率可达 85-90%"
    },
    "cifar100": {
        "name": "CIFAR-100",
        "full_name": "Canadian Institute For Advanced Research - 100 Classes",
        "description": "CIFAR-100 是 CIFAR-10 的扩展版，包含 60,000 张 32×32 RGB 彩色图像，但分为 100 个细粒度类别（如各种鸟类、各种花卉）。每个类别只有 600 张图片，对模型的泛化能力和特征提取提出了更高要求。",
        "difficulty": "高级",
        "paper": "Krizhevsky, 2009",
        "classes": ["100 个细分类别（如各种动物、植物、交通工具等）"],
        "channels": "RGB (3通道)",
        "recommended_model": "深度网络 (4-6 层卷积) + 数据增强 + 正则化，准确率可达 60-70%"
    }
}

# Default network architectures per dataset
DEFAULT_NETWORKS = {
    "mnist": {
        "layers": [
            {"type": "conv", "name": "Conv1", "out_channels": 32, "kernel_size": 3, "padding": 1, "activation": "relu"},
            {"type": "pool", "name": "Pool1", "pool_size": 2, "stride": 2},
            {"type": "conv", "name": "Conv2", "out_channels": 64, "kernel_size": 3, "padding": 1, "activation": "relu"},
            {"type": "pool", "name": "Pool2", "pool_size": 2, "stride": 2},
            {"type": "flatten", "name": "Flatten"},
            {"type": "fc", "name": "FC1", "units": 128, "activation": "relu"},
            {"type": "fc", "name": "FC2", "units": 10},
        ],
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 10
    },
    "fashion_mnist": {
        "layers": [
            {"type": "conv", "name": "Conv1", "out_channels": 32, "kernel_size": 3, "padding": 1, "activation": "relu"},
            {"type": "conv", "name": "Conv2", "out_channels": 64, "kernel_size": 3, "padding": 1, "activation": "relu"},
            {"type": "pool", "name": "Pool1", "pool_size": 2, "stride": 2},
            {"type": "conv", "name": "Conv3", "out_channels": 128, "kernel_size": 3, "padding": 1, "activation": "relu"},
            {"type": "pool", "name": "Pool2", "pool_size": 2, "stride": 2},
            {"type": "flatten", "name": "Flatten"},
            {"type": "fc", "name": "FC1", "units": 256, "activation": "relu"},
            {"type": "fc", "name": "FC2", "units": 10},
        ],
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 20
    },
    "cifar10": {
        "layers": [
            {"type": "conv", "name": "Conv1", "out_channels": 64, "kernel_size": 3, "padding": 1, "activation": "relu"},
            {"type": "conv", "name": "Conv2", "out_channels": 64, "kernel_size": 3, "padding": 1, "activation": "relu"},
            {"type": "pool", "name": "Pool1", "pool_size": 2, "stride": 2},
            {"type": "conv", "name": "Conv3", "out_channels": 128, "kernel_size": 3, "padding": 1, "activation": "relu"},
            {"type": "conv", "name": "Conv4", "out_channels": 128, "kernel_size": 3, "padding": 1, "activation": "relu"},
            {"type": "pool", "name": "Pool2", "pool_size": 2, "stride": 2},
            {"type": "flatten", "name": "Flatten"},
            {"type": "fc", "name": "FC1", "units": 512, "activation": "relu"},
            {"type": "fc", "name": "FC2", "units": 10},
        ],
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 30
    },
    "cifar100": {
        "layers": [
            {"type": "conv", "name": "Conv1", "out_channels": 64, "kernel_size": 3, "padding": 1, "activation": "relu"},
            {"type": "conv", "name": "Conv2", "out_channels": 64, "kernel_size": 3, "padding": 1, "activation": "relu"},
            {"type": "pool", "name": "Pool1", "pool_size": 2, "stride": 2},
            {"type": "conv", "name": "Conv3", "out_channels": 128, "kernel_size": 3, "padding": 1, "activation": "relu"},
            {"type": "conv", "name": "Conv4", "out_channels": 128, "kernel_size": 3, "padding": 1, "activation": "relu"},
            {"type": "pool", "name": "Pool2", "pool_size": 2, "stride": 2},
            {"type": "conv", "name": "Conv5", "out_channels": 256, "kernel_size": 3, "padding": 1, "activation": "relu"},
            {"type": "pool", "name": "Pool3", "pool_size": 2, "stride": 2},
            {"type": "flatten", "name": "Flatten"},
            {"type": "fc", "name": "FC1", "units": 512, "activation": "relu"},
            {"type": "fc", "name": "FC2", "units": 100},
        ],
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 50
    }
}

class CNNLayer(BaseModel):
    type: Literal["conv", "pool", "fc", "flatten"]
    name: str
    in_channels: Optional[int] = None
    out_channels: Optional[int] = Field(None, gt=0)
    kernel_size: Optional[int] = Field(None, gt=0)
    padding: Optional[int] = Field(None, ge=0)
    pool_size: Optional[int] = Field(None, gt=0)
    stride: Optional[int] = Field(None, gt=0)
    units: Optional[int] = Field(None, gt=0)
    activation: Optional[Literal["relu", "softmax", "sigmoid"]] = None

class ModelConfig(BaseModel):
    layers: List[CNNLayer]
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    learning_rate: float = Field(0.001, gt=0, le=1.0)

DatasetName = Literal["mnist", "fashion_mnist", "cifar10", "cifar100"]

class BatchTrainingRequest(BaseModel):
    tenant_id: str
    batch_size: int = Field(32, gt=0, le=1024)
    learning_rate: float = Field(0.001, gt=0, le=1.0)
    dataset: DatasetName = "mnist"
    epochs: int = Field(1, gt=0, le=100)

class TrainingRequest(BaseModel):
    tenant_id: str
    batch_size: int = Field(32, gt=0, le=1024)
    learning_rate: float = Field(0.001, gt=0, le=1.0)
    dataset: DatasetName = "mnist"
    epochs: int = Field(1, gt=0, le=100)

class InferenceRequest(BaseModel):
    tenant_id: str
    image_data: List[float] = Field(..., min_length=1)

class CreateModelRequest(BaseModel):
    tenant_id: str
    config: ModelConfig
    dataset: DatasetName = "mnist"

# Global train loaders cache
train_loaders: Dict[str, torch.utils.data.DataLoader] = {}
val_loaders: Dict[str, torch.utils.data.DataLoader] = {}

# Dataset class mapping
DATASET_CLASSES = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
}

def _load_torchvision_dataset(dataset_name: str, train: bool, transform):
    """Load a torchvision dataset. Try offline first, fall back to download."""
    ds_cls = DATASET_CLASSES.get(dataset_name)
    if ds_cls is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    import os
    # Use absolute path and ensure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Try offline first (data bundled in project)
    try:
        return ds_cls(data_dir, train=train, download=False, transform=transform)
    except (RuntimeError, FileNotFoundError, Exception) as e:
        print(f"[Dataset] Offline load failed for {dataset_name}: {type(e).__name__}")

    # Fall back to download
    print(f"[Dataset] Downloading {dataset_name} (train={train}) to {data_dir}...")
    try:
        return ds_cls(data_dir, train=train, download=True, transform=transform)
    except Exception as e:
        print(f"[Dataset] Download failed for {dataset_name}: {e}")
        raise

def get_train_loader(dataset_name: str, batch_size: int, tenant_id: str = None):
    """Get DataLoader for a specific tenant to ensure isolation."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cache_key = f"{dataset_name}_{batch_size}_{tenant_id or 'default'}"
    if cache_key in train_loaders:
        return train_loaders[cache_key]

    config = DATASET_CONFIGS[dataset_name]
    dataset = _load_torchvision_dataset(dataset_name, train=True, transform=config["transform"])

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True if device.type != "cpu" else False
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
    dataset = _load_torchvision_dataset(dataset_name, train=False, transform=config["transform"])

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
        self.activation_funcs = []
        self.has_flatten = False

        prev_channels = dataset_config["input_channels"]
        prev_h = dataset_config["image_size"]
        prev_w = dataset_config["image_size"]
        flat_size = None  # Will be set when flatten is encountered

        for layer in config.layers:
            if layer.type == 'conv':
                k = layer.kernel_size or 3
                p = layer.padding or 0
                s = 1  # stride=1 for conv
                out_ch = layer.out_channels or 32

                self.layers_list.append(nn.Conv2d(
                    prev_channels, out_ch, k, stride=s, padding=p
                ))

                if layer.activation == 'relu':
                    self.activation_funcs.append(nn.ReLU())
                else:
                    self.activation_funcs.append(None)

                # Track spatial dimensions: output = floor((input + 2*pad - kernel) / stride) + 1
                prev_h = (prev_h + 2 * p - k) // s + 1
                prev_w = (prev_w + 2 * p - k) // s + 1
                prev_channels = out_ch

            elif layer.type == 'pool':
                ps = layer.pool_size or 2
                st = layer.stride or ps  # default stride = pool_size

                self.layers_list.append(nn.MaxPool2d(ps, st))
                self.activation_funcs.append(None)

                prev_h = prev_h // ps
                prev_w = prev_w // ps

            elif layer.type == 'flatten':
                self.has_flatten = True
                flat_size = prev_channels * prev_h * prev_w
                # flatten is handled in forward(), not added to layers_list
                # so no activation_funcs entry either

            elif layer.type == 'fc':
                in_features = flat_size if flat_size is not None else prev_channels
                units = layer.units or 128

                self.layers_list.append(nn.Linear(in_features, units))

                if layer.activation == 'relu':
                    self.activation_funcs.append(nn.ReLU())
                else:
                    self.activation_funcs.append(None)

                prev_channels = units
                flat_size = None  # subsequent FC layers chain from previous

    def forward(self, x):
        flatten_done = False
        for i, layer in enumerate(self.layers_list):
            # Apply flatten before first FC layer
            if isinstance(layer, nn.Linear) and self.has_flatten and not flatten_done:
                if x.dim() > 2:
                    x = x.flatten(1)
                flatten_done = True

            x = layer(x)

            activation = self.activation_funcs[i]
            if activation is not None:
                x = activation(x)

        return x

def create_model_for_tenant(tenant_id: str, config: ModelConfig, dataset: str):
    """Create a new model for a tenant on the current device."""
    model = ConfigurableCNN(config, dataset).to(device)
    print(f"Created model for tenant '{tenant_id}' on {device}")

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
    # Preload small datasets only (MNIST ~10MB, Fashion-MNIST ~30MB, KMNIST ~10MB)
    # Large datasets (CIFAR-10 ~170MB, CIFAR-100 ~170MB) will be downloaded on first use
    SMALL_DATASETS = ["mnist", "fashion_mnist"]
    for ds_name in SMALL_DATASETS:
        try:
            get_train_loader(ds_name, 32)
            print(f"Preloaded {ds_name} dataset")
        except Exception as e:
            print(f"Warning: Failed to preload {ds_name}: {e}")
    yield
    # Cleanup on shutdown
    print("Shutting down...")
    # Clear all caches on shutdown
    tenant_states.clear()
    train_loaders.clear()
    val_loaders.clear()
    print("All caches cleared")

app = FastAPI(title="CNN Training View API", lifespan=lifespan)

# Strict CORS configuration - only allow local development origins
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",  # Vite default
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
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
    if not validate_tenant_id(request.tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    if len(tenant_states) >= MAX_TENANTS and request.tenant_id not in tenant_states:
        raise HTTPException(status_code=429, detail=f"Maximum number of tenants ({MAX_TENANTS}) reached")
    try:
        # Log the actual config received with full detail
        for i, l in enumerate(request.config.layers):
            detail = {k: v for k, v in l.model_dump().items() if v is not None}
            print(f"  Layer {i}: {detail}")
        print(f"  optimizer={request.config.optimizer} lr={request.config.learning_rate} dataset={request.dataset}")

        model = create_model_for_tenant(request.tenant_id, request.config, request.dataset)
        # Log model architecture summary
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model created: {len(model.layers_list)} layers, {total_params} params")
        return {
            "tenant_id": request.tenant_id,
            "status": "created",
            "message": f"Model created for tenant {request.tenant_id}"
        }
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail="Invalid model configuration")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to create model")

@app.post("/api/train")
async def train(request: TrainingRequest):
    # Log which model is being trained
    if request.tenant_id in tenant_states:
        state = tenant_states[request.tenant_id]
        model = state["model"]
        layer_types = [type(l).__name__ for l in model.layers_list]
        print(f"[train] tenant={request.tenant_id} model_layers={layer_types} dataset={request.dataset}")
    result = await train_epochs(request.tenant_id, request.batch_size, request.learning_rate, request.dataset, 1)
    # Return single-epoch format for backward compatibility
    entry = result["results"][0]
    return {
        "tenant_id": result["tenant_id"],
        "epoch": entry["epoch"],
        "loss": entry["loss"],
        "accuracy": entry["accuracy"],
        "layer_activations": result["layer_activations"]
    }

@app.post("/api/train/batch")
async def train_batch(request: BatchTrainingRequest):
    """Train multiple epochs in a single request."""
    return await train_epochs(request.tenant_id, request.batch_size, request.learning_rate, request.dataset, request.epochs)

def _run_training(state: dict, train_loader, epochs: int, dataset: str) -> dict:
    """Synchronous training function, runs in thread pool."""
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    criterion = state["criterion"]

    model.train()

    all_results = []
    layer_activations = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        batches_processed = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            batches_processed += 1

        scheduler.step()

        state["current_epoch"] += 1
        avg_loss = total_loss / max(batches_processed, 1)
        accuracy = correct / max(total, 1)

        state["training_history"].append({
            "epoch": state["current_epoch"],
            "loss": round(avg_loss, 4),
            "accuracy": round(accuracy, 4)
        })

        all_results.append({
            "epoch": state["current_epoch"],
            "loss": round(avg_loss, 4),
            "accuracy": round(accuracy, 4)
        })

    # Get final layer activations
    for layer in model.layers_list:
        if isinstance(layer, nn.Conv2d):
            layer_activations.append(layer.weight.abs().mean().item())

    return {
        "epochs_trained": epochs,
        "start_epoch": state["current_epoch"] - epochs + 1,
        "end_epoch": state["current_epoch"],
        "results": all_results,
        "layer_activations": layer_activations
    }

async def train_epochs(tenant_id: str, batch_size: int, learning_rate: float, dataset: str, epochs: int):
    """Train multiple epochs without blocking the event loop."""
    if not validate_tenant_id(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")

    if tenant_id not in tenant_states:
        raise HTTPException(status_code=404, detail=f"No model found for tenant {tenant_id}. Create one first via POST /api/model/create")

    lock = get_tenant_lock(tenant_id)
    if lock.locked():
        raise HTTPException(status_code=409, detail="Training already in progress for this tenant")

    async with lock:
        state = tenant_states[tenant_id]
        train_loader = get_train_loader(dataset, batch_size, tenant_id)

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                _run_training,
                state, train_loader, epochs, dataset
            )
        except RuntimeError as e:
            print(f"[train_epochs] RuntimeError: {e}")
            raise HTTPException(status_code=500, detail="Training error: check model configuration")
        except Exception as e:
            print(f"[train_epochs] Exception: {type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail="Training failed")

    result["tenant_id"] = tenant_id
    return result

@app.post("/api/inference")
async def inference(request: InferenceRequest):
    tenant_id = request.tenant_id

    if not validate_tenant_id(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")

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
            # Grayscale: single channel
            img_array = torch.tensor(request.image_data).reshape(1, 1, image_size, image_size)
        elif len(request.image_data) == image_size * image_size * channels:
            # RGB: multiple channels
            img_array = torch.tensor(request.image_data).reshape(1, channels, image_size, image_size)
        else:
            raise HTTPException(status_code=400, detail="Invalid image dimensions")

        # Normalize with same parameters as training
        norm_config = DATASET_CONFIGS[dataset]["transform"].transforms[1]
        mean = torch.tensor(norm_config.mean).reshape(1, -1, 1, 1)
        std = torch.tensor(norm_config.std).reshape(1, -1, 1, 1)
        img_tensor = img_array.float().to(device)
        img_tensor = (img_tensor - mean.to(device)) / std.to(device)

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
    except HTTPException:
        raise
    except (ValueError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid input data for inference")
    except Exception:
        raise HTTPException(status_code=500, detail="Inference failed")

@app.get("/api/model/summary/{tenant_id}")
async def model_summary(tenant_id: str):
    if not validate_tenant_id(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
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
    result = []
    for name, config in DATASET_CONFIGS.items():
        desc = DATASET_DESCRIPTIONS.get(name, {})
        result.append({
            "name": name,
            "description": desc.get("description", "").split("。")[0] if desc.get("description") else name,
            "classes": config["num_classes"],
            "image_size": config["image_size"],
            "channels": config["input_channels"],
            "difficulty": desc.get("difficulty", ""),
            "full_name": desc.get("full_name", name),
            "class_list": desc.get("classes", []),
            "recommended_model": desc.get("recommended_model", ""),
            "paper": desc.get("paper", "")
        })
    return {"datasets": result}


@app.get("/api/datasets/{dataset_name}/info")
async def get_dataset_info(dataset_name: str):
    if dataset_name not in DATASET_DESCRIPTIONS:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    desc = DATASET_DESCRIPTIONS[dataset_name]
    config = DATASET_CONFIGS[dataset_name]
    return {
        **desc,
        "image_size": config["image_size"],
        "input_channels": config["input_channels"],
        "num_classes": config["num_classes"],
        "train_samples": 60000 if dataset_name in ("mnist", "fashion_mnist") else 50000,
        "test_samples": 10000
    }


@app.get("/api/datasets/{dataset_name}/default-network")
async def get_default_network(dataset_name: str):
    if dataset_name not in DEFAULT_NETWORKS:
        raise HTTPException(status_code=404, detail=f"No default network for {dataset_name}")
    return DEFAULT_NETWORKS[dataset_name]

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
        if dataset_name in ("mnist", "fashion_mnist"):
            # Grayscale datasets: data shape [batch, 1, 28, 28]
            config = DATASET_CONFIGS[dataset_name]
            mean = config["transform"].transforms[1].mean[0]
            std = config["transform"].transforms[1].std[0]
            img_array = (data.numpy() * std + mean).clip(0, 1) * 255
            img_array = img_array.astype(np.uint8)
            for j in range(min(count, data.shape[0])):
                samples.append(img_array[j].flatten().tolist())
                labels.append(int(target[j]))
        elif dataset_name in ("cifar10", "cifar100"):
            # RGB datasets: data shape [batch, 3, 32, 32]
            config = DATASET_CONFIGS[dataset_name]
            mean = np.array(config["transform"].transforms[1].mean).reshape(1, 3, 1, 1)
            std = np.array(config["transform"].transforms[1].std).reshape(1, 3, 1, 1)
            img_array = ((data.numpy() * std + mean).clip(0, 1) * 255).astype(np.uint8)
            # Convert from CHW to HWC for grayscale-style display
            # Keep as CHW for consistency with inference format
            for j in range(min(count, data.shape[0])):
                samples.append(img_array[j].flatten().tolist())
                labels.append(int(target[j]))
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
    if not validate_tenant_id(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    if tenant_id in tenant_states:
        # Clean up DataLoader cache for this tenant
        keys_to_remove = [k for k in train_loaders.keys() if tenant_id in k]
        for k in keys_to_remove:
            del train_loaders[k]
        del tenant_states[tenant_id]
        return {"status": "deleted", "tenant_id": tenant_id}
    raise HTTPException(status_code=404, detail=f"No model found for tenant {tenant_id}")

@app.post("/api/cache/cleanup")
async def cleanup_cache():
    """Manually trigger cleanup of stale caches."""
    before_tenants = len(tenant_states)
    before_loaders = len(train_loaders)

    # Clean up old tenant states (inactive for more than 1 hour)
    current_time = time.time()
    stale_tenants = [
        tid for tid, state in tenant_states.items()
        if current_time - state.get("created_at", 0) > 3600
    ]
    for tid in stale_tenants:
        del tenant_states[tid]
        # Clean up loaders for this tenant
        keys_to_remove = [k for k in train_loaders.keys() if tid in k]
        for k in keys_to_remove:
            del train_loaders[k]

    return {
        "status": "cleaned",
        "tenants_removed": len(stale_tenants),
        "remaining_tenants": len(tenant_states),
        "loaders_cleaned": before_loaders - len(train_loaders)
    }

@app.get("/api/cache/stats")
async def cache_stats():
    """Get current cache statistics."""
    return {
        "active_tenants": len(tenant_states),
        "cached_train_loaders": len(train_loaders),
        "cached_val_loaders": len(val_loaders),
        "train_loader_keys": list(train_loaders.keys()),
        "val_loader_keys": list(val_loaders.keys())
    }

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