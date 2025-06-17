import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import logging
from datetime import datetime
import os
from sklearn.metrics import confusion_matrix


# 配置类
class Config:
    def __init__(self):
        # 模型架构参数
        self.num_experts = 4
        self.expert_hidden = 128
        self.task_embed_dim = 16
        self.adapter_bottleneck_dim = 64
        self.num_tasks = 5

        # 训练超参数
        self.pretrain_epochs = 5
        self.task_epochs = 1
        self.batch_size = 64
        self.pretrain_lr = 0.0005
        self.adapter_lr = 0.01
        self.lr_patience = 2
        self.lr_factor = 0.5
        self.grad_clip_norm = 5.0

        # 优化器参数
        self.optimizer_type = "Adam"  # Adam 或 SGD
        self.momentum = 0.9  # SGD 动量
        self.weight_decay = 1e-5

        # 数据处理参数
        self.use_weighted_sampler = True
        self.data_augmentation = False

        # 日志和保存参数
        self.log_level = "DEBUG"  # DEBUG 或 INFO
        self.checkpoint_path = os.path.join("checkpoints", "mmoe_weights.pth")
        self.save_frequency = 0  # 每 N 个 epoch 保存检查点，0 表示仅保存最终权重
        self.tensorboard_dir = os.path.join("runs", datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.pretrained = True

    def log_config(self, logger):
        logger.info("实验配置参数：")
        for attr, value in vars(self).items():
            logger.info(f"  {attr}: {value}")
        self.logger = logger


# 设置日志
def setup_logging(log_level):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


# MMoE 模型
class MMOE(nn.Module):
    def __init__(self, config):
        super(MMOE, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        # 动态计算特征维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)
            dummy_output = self.feature_extractor(dummy_input)
            self.feature_dim = dummy_output.shape[1]
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, config.expert_hidden),
                nn.BatchNorm1d(config.expert_hidden),
                nn.LeakyReLU(0.1),
                nn.Linear(config.expert_hidden, config.expert_hidden),
                nn.BatchNorm1d(config.expert_hidden),
                nn.LeakyReLU(0.1)
            ) for _ in range(config.num_experts)
        ])
        self.task_embedding = nn.Embedding(config.num_tasks, config.task_embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.feature_dim + config.task_embed_dim, config.num_experts),
            nn.Softmax(dim=-1)
        )
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.task_embedding.weight, mean=0, std=0.01)

    def forward(self, x, task_id):
        if not isinstance(task_id, torch.Tensor):
            task_id = torch.tensor([task_id], dtype=torch.long, device=x.device)
        features = self.feature_extractor(x)
        task_embed = self.task_embedding(task_id)
        task_embed = task_embed.expand(features.size(0), -1)
        gate_input = torch.cat([features, task_embed], dim=1)
        gate_weights = self.gate(gate_input)
        expert_outputs = [expert(features) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        weighted_output = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)
        return weighted_output, gate_weights


# 适配器模块
class Adapter(nn.Module):
    def __init__(self, in_dim, num_classes, bottleneck_dim):
        super(Adapter, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(bottleneck_dim, num_classes)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# 持续学习模型
class ContinualLearner(nn.Module):
    def __init__(self, config):
        super(ContinualLearner, self).__init__()
        self.mmoe = MMOE(config)
        self.adapters = nn.ModuleDict()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.to(self.device)

    def add_task(self, task_id, num_classes=2):
        expert = self.mmoe.experts[0]
        in_dim = expert[-3].out_features  # 直接访问最后一个 Linear 层
        adapter = Adapter(in_dim, num_classes, self.config.adapter_bottleneck_dim).to(self.device)
        self.adapters[str(task_id)] = adapter
        self.config.logger.info(f"任务 {task_id} 添加适配器，输入维度: {in_dim}, 输出维度: {num_classes}")

    def forward(self, x, task_id):
        x = x.to(self.device)
        task_id_tensor = torch.tensor([task_id], dtype=torch.long, device=self.device)
        shared_output, gate_weights = self.mmoe(x, task_id_tensor)
        adapter_key = str(task_id)
        if adapter_key in self.adapters:
            output = self.adapters[adapter_key](shared_output)
        else:
            raise ValueError(f"未找到任务 {task_id} 的适配器")
        return output, gate_weights


# 数据集准备
def get_split_mnist(task_id, config):
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    if config.data_augmentation:
        transform_list.insert(0, transforms.RandomRotation(10))
    transform = transforms.Compose(transform_list)

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    class_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    if task_id >= len(class_pairs):
        raise ValueError(f"任务 ID {task_id} 超出最大任务数 {len(class_pairs)}")
    classes = class_pairs[task_id]
    train_indices = [i for i in range(len(train_dataset)) if train_dataset.targets[i] in classes]
    test_indices = [i for i in range(len(test_dataset)) if test_dataset.targets[i] in classes]
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    class RemappedDataset:
        def __init__(self, subset, classes):
            self.subset = subset
            self.classes = classes

        def __getitem__(self, idx):
            data, target = self.subset[idx]
            for i, cls in enumerate(self.classes):
                if target == cls:
                    target = i
                    break
            return data, target

        def __len__(self):
            return len(self.subset)

    remapped_train_dataset = RemappedDataset(train_subset, classes)
    remapped_test_dataset = RemappedDataset(test_subset, classes)

    if config.use_weighted_sampler:
        train_labels = np.array([target for _, target in remapped_train_dataset])
        class_counts = np.bincount(train_labels)
        weights = np.array([1.0 / class_counts[label] for label in train_labels])
        sampler = WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(remapped_train_dataset, batch_size=config.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(remapped_train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(remapped_test_dataset, batch_size=config.batch_size, shuffle=False)

    test_labels = np.array([target for _, target in remapped_test_dataset])
    config.logger.info(f"任务 {task_id} 训练标签分布: {np.bincount(train_labels)}")
    config.logger.info(f"任务 {task_id} 测试标签分布: {np.bincount(test_labels)}")
    data, _ = next(iter(train_loader))
    config.logger.debug(
        f"任务 {task_id} 输入数据形状: {data.shape}, 值范围: [{data.min().item()}, {data.max().item()}], 均值: {data.mean().item():.4f}, 标准差: {data.std().item():.4f}")

    return train_loader, test_loader, classes


# 预训练 MMoE
def pretrain_mmoe(model, train_loaders, config):
    with SummaryWriter(log_dir=f"{config.tensorboard_dir}/pretrain") as writer:
        if os.path.exists(config.checkpoint_path) and config.pretrained:
            try:
                config.logger.info(f"找到预训练权重文件 {config.checkpoint_path}，尝试加载")
                model.mmoe.load_state_dict(torch.load(config.checkpoint_path, map_location=model.device))
                config.logger.info(f"成功加载预训练权重 {config.checkpoint_path}")
                return
            except Exception as e:
                config.logger.error(f"加载预训练权重失败: {e}，将重新预训练")

        optimizer = optim.Adam(model.mmoe.parameters(), lr=config.pretrain_lr,
                              weight_decay=config.weight_decay) if config.optimizer_type == "Adam" else optim.SGD(
            model.mmoe.parameters(), lr=config.pretrain_lr, momentum=config.momentum, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor,
                                                         patience=config.lr_patience)

        config.logger.info(
            f"开始预训练 MMoE，任务数: {len(train_loaders)}, Epochs: {config.pretrain_epochs}, 优化器: {config.optimizer_type}, 学习率: {config.pretrain_lr}")

        # 初始化临时头部
        temp_heads = {
            task_id: nn.Linear(model.mmoe.experts[0][-3].out_features, 2).to(model.device)
            for task_id in range(len(train_loaders))
        }
        for _, head in temp_heads.items():
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

        for epoch in range(config.pretrain_epochs):
            model.train()
            total_loss, correct, total = 0, 0, 0
            epoch_start = time.time()
            all_preds, all_targets = [], []

            for task_id, train_loader in enumerate(train_loaders):
                temp_head = temp_heads[task_id]
                train_labels = np.concatenate([target.cpu().numpy() for _, target in train_loader])
                class_counts = np.bincount(train_labels)
                class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float).to(model.device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                temp_optimizer = optim.Adam([
                    {'params': model.mmoe.parameters(), 'lr': config.pretrain_lr, 'weight_decay': config.weight_decay},
                    {'params': temp_head.parameters(), 'lr': config.pretrain_lr * 2, 'weight_decay': config.weight_decay}
                ]) if config.optimizer_type == "Adam" else optim.SGD([
                    {'params': model.mmoe.parameters(), 'lr': config.pretrain_lr, 'momentum': config.momentum,
                     'weight_decay': config.weight_decay},
                    {'params': temp_head.parameters(), 'lr': config.pretrain_lr * 2, 'momentum': config.momentum,
                     'weight_decay': config.weight_decay}
                ])
                config.logger.info(f"任务 {task_id} 训练标签分布: {class_counts}, 类权重: {class_weights.cpu().numpy()}")

                pbar = tqdm(train_loader, desc=f'Pretrain Task {task_id} Epoch {epoch + 1}/{config.pretrain_epochs}')
                for batch_idx, (data, target) in enumerate(pbar):
                    data, target = data.to(model.device), target.to(model.device)
                    task_id_tensor = torch.tensor([task_id], dtype=torch.long, device=model.device)

                    temp_optimizer.zero_grad()
                    output, gate_weights = model.mmoe(data, task_id_tensor)
                    output = temp_head(output)
                    loss = criterion(output, target)

                    loss.backward()
                    grad_norm = sum(p.grad.norm().item() for p in model.mmoe.parameters() if p.grad is not None)
                    temp_head_grad_norm = sum(p.grad.norm().item() for p in temp_head.parameters() if p.grad is not None)
                    if batch_idx % 10 == 0:  # 减少日志频率
                        config.logger.debug(
                            f"任务 {task_id}, Epoch {epoch + 1}, Batch {batch_idx + 1}, MMoE 输出均值: {output.mean().item():.4f}, 标准差: {output.std().item():.4f}")
                        config.logger.debug(
                            f"任务 {task_id}, Epoch {epoch + 1}, Batch {batch_idx + 1}, Gate 权重均值: {gate_weights.mean().item():.4f}, 标准差: {gate_weights.std().item():.4f}")
                        config.logger.debug(
                            f"任务 {task_id}, Epoch {epoch + 1}, Batch {batch_idx + 1}, MMoE 梯度范数: {grad_norm:.4f}, Temp Head 梯度范数: {temp_head_grad_norm:.4f}")
                        writer.add_histogram(f"Gradient_Norm/Task_{task_id}/MMoE", grad_norm,
                                             epoch * len(train_loader) + batch_idx)
                        writer.add_histogram(f"Gradient_Norm/Task_{task_id}/Temp_Head", temp_head_grad_norm,
                                             epoch * len(train_loader) + batch_idx)
                        writer.add_histogram(f"Activation/Task_{task_id}/MMoE_Output", output,
                                             epoch * len(train_loader) + batch_idx)
                        writer.add_histogram(f"Gate_Weights/Task_{task_id}", gate_weights,
                                             epoch * len(train_loader) + batch_idx)

                    torch.nn.utils.clip_grad_norm_(list(model.mmoe.parameters()) + list(temp_head.parameters()),
                                                   max_norm=config.grad_clip_norm)
                    temp_optimizer.step()

                    if batch_idx % 10 == 0:
                        for name, param in model.mmoe.named_parameters():
                            if param.grad is not None:
                                writer.add_histogram(f"Weight_Update/Task_{task_id}/{name}", param.grad,
                                                     epoch * len(train_loader) + batch_idx)

                    total_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    batch_acc = 100. * (predicted == target).sum().item() / target.size(0)
                    writer.add_scalar(f"Train_Loss/Task_{task_id}/Batch", loss.item(),
                                      epoch * len(train_loader) + batch_idx)
                    writer.add_scalar(f"Train_Accuracy/Task_{task_id}/Batch", batch_acc,
                                      epoch * len(train_loader) + batch_idx)
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_acc:.2f}%'})

            avg_loss = total_loss / sum(len(loader) for loader in train_loaders)
            train_acc = 100. * correct / total
            cm = confusion_matrix(all_targets, all_preds)
            config.logger.info(f"预训练 Epoch {epoch + 1}/{config.pretrain_epochs}, 训练混淆矩阵:\n{cm}")

            model.eval()
            test_accs, test_losses = [], []
            for task_id in range(len(train_loaders)):
                _, test_loader, classes = get_split_mnist(task_id, config)
                test_correct, test_total, test_loss = 0, 0, 0
                test_preds, test_targets = [], []
                temp_head = temp_heads[task_id]
                criterion = nn.CrossEntropyLoss()
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(model.device), target.to(model.device)
                        task_id_tensor = torch.tensor([task_id], dtype=torch.long, device=model.device)
                        output, _ = model.mmoe(data, task_id_tensor)
                        output = temp_head(output)
                        loss = criterion(output, target)
                        test_loss += loss.item()
                        _, predicted = torch.max(output.data, 1)
                        test_total += target.size(0)
                        test_correct += (predicted == target).sum().item()
                        test_preds.extend(predicted.cpu().numpy())
                        test_targets.extend(target.cpu().numpy())
                test_acc = 100. * test_correct / test_total
                test_loss = test_loss / len(test_loader)
                test_accs.append(test_acc)
                test_losses.append(test_loss)
                test_cm = confusion_matrix(test_targets, test_preds)
                config.logger.info(
                    f"预训练 Epoch {epoch + 1}/{config.pretrain_epochs}, 任务 {task_id} 测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%, 测试混淆矩阵:\n{test_cm}")
                writer.add_scalar(f"Test_Loss/Task_{task_id}", test_loss, epoch + 1)
                writer.add_scalar(f"Test_Accuracy/Task_{task_id}", test_acc, epoch + 1)

            avg_test_acc = sum(test_accs) / len(test_accs)
            avg_test_loss = sum(test_losses) / len(test_losses)
            writer.add_scalar("Train_Loss/Epoch", avg_loss, epoch + 1)
            writer.add_scalar("Train_Accuracy/Epoch", train_acc, epoch + 1)
            writer.add_scalar("Test_Loss/Average", avg_test_loss, epoch + 1)
            writer.add_scalar("Test_Accuracy/Average", avg_test_acc, epoch + 1)
            writer.add_scalar("Learning_Rate/MMoE", optimizer.param_groups[0]['lr'], epoch + 1)

            epoch_time = time.time() - epoch_start
            config.logger.info(
                f"预训练 Epoch {epoch + 1}/{config.pretrain_epochs}, 训练损失: {avg_loss:.4f}, 训练准确率: {train_acc:.2f}%, 平均测试损失: {avg_test_loss:.4f}, 平均测试准确率: {avg_test_acc:.2f}%, 学习率: {optimizer.param_groups[0]['lr']}, 时间: {epoch_time:.2f}s"
            )
            scheduler.step(avg_loss)

            if config.save_frequency > 0 and (epoch + 1) % config.save_frequency == 0:
                checkpoint_file = f"{os.path.splitext(config.checkpoint_path)[0]}_epoch_{epoch + 1}.pth"
                torch.save(model.mmoe.state_dict(), checkpoint_file)
                config.logger.info(f"保存预训练检查点: {checkpoint_file}")

        os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
        torch.save(model.mmoe.state_dict(), config.checkpoint_path)
        config.logger.info(f"MMoE 预训练完成，权重已保存至 {config.checkpoint_path}")


# 训练新任务
def train_task(model, task_id, train_loader, test_loader, classes, config):
    with SummaryWriter(log_dir=f"{config.tensorboard_dir}/task_{task_id}") as writer:
        model.add_task(task_id, num_classes=len(classes))
        train_labels = np.concatenate([target.cpu().numpy() for _, target in train_loader])
        class_counts = np.bincount(train_labels)
        class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float).to(model.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.adapters.parameters(), lr=config.adapter_lr,
                              weight_decay=config.weight_decay) if config.optimizer_type == "Adam" else optim.SGD(
            model.adapters.parameters(), lr=config.adapter_lr, momentum=config.momentum, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor,
                                                         patience=config.lr_patience)

        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        config.logger.info(
            f"开始训练任务 {task_id}，类别: {classes}, Epochs: {config.task_epochs}, 训练样本: {len(train_loader.dataset)}, 测试样本: {len(test_loader.dataset)}, 优化器: {config.optimizer_type}, 学习率: {config.adapter_lr}, 类权重: {class_weights.cpu().numpy()}"
        )

        for epoch in range(config.task_epochs):
            model.train()
            total_loss, correct, total = 0, 0, 0
            epoch_start = time.time()
            all_preds, all_targets = [], []
            pbar = tqdm(train_loader, desc=f'Task {task_id} Epoch {epoch + 1}/{config.task_epochs}')

            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(model.device), target.to(model.device)
                if batch_idx == 0:
                    config.logger.debug(
                        f"任务 {task_id}, Epoch {epoch + 1}, Batch 标签样本: {target[:5].cpu().numpy()}, 分布: {np.bincount(target.cpu().numpy())}")

                optimizer.zero_grad()
                output, gate_weights = model(data, task_id)
                loss = criterion(output, target)
                loss.backward()

                grad_norm = sum(p.grad.norm().item() for p in model.adapters.parameters() if p.grad is not None)
                if batch_idx % 10 == 0:
                    config.logger.debug(f"任务 {task_id}, Epoch {epoch + 1}, Batch {batch_idx + 1}, 适配器梯度范数: {grad_norm:.4f}")
                    writer.add_histogram(f"Gradient_Norm/Task_{task_id}/Adapter", grad_norm,
                                         epoch * len(train_loader) + batch_idx)
                    writer.add_histogram(f"Activation/Task_{task_id}/Output", output,
                                         epoch * len(train_loader) + batch_idx)
                    writer.add_histogram(f"Gate_Weights/Task_{task_id}", gate_weights,
                                         epoch * len(train_loader) + batch_idx)

                torch.nn.utils.clip_grad_norm_(model.adapters.parameters(), max_norm=config.grad_clip_norm)
                optimizer.step()

                if batch_idx % 10 == 0:
                    for name, param in model.adapters.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(f"Weight_Update/Task_{task_id}/{name}", param.grad,
                                                 epoch * len(train_loader) + batch_idx)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                batch_acc = 100. * (predicted == target).sum().item() / target.size(0)
                writer.add_scalar(f"Train_Loss/Task_{task_id}/Batch", loss.item(),
                                  epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f"Train_Accuracy/Task_{task_id}/Batch", batch_acc,
                                  epoch * len(train_loader) + batch_idx)
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_acc:.2f}%'})

            avg_loss = total_loss / len(train_loader)
            train_acc = 100. * correct / total
            cm = confusion_matrix(all_targets, all_preds)
            config.logger.info(f"任务 {task_id} Epoch {epoch + 1}/{config.task_epochs}, 训练混淆矩阵:\n{cm}")

            model.eval()
            test_correct, test_total, test_loss = 0, 0, 0
            test_preds, test_targets = [], []
            test_criterion = nn.CrossEntropyLoss()
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(model.device), target.to(model.device)
                    output, _ = model(data, task_id)
                    loss = test_criterion(output, target)
                    test_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    test_total += target.size(0)
                    test_correct += (predicted == target).sum().item()
                    test_preds.extend(predicted.cpu().numpy())
                    test_targets.extend(target.cpu().numpy())
            test_acc = 100. * test_correct / test_total
            test_loss = test_loss / len(test_loader)
            test_cm = confusion_matrix(test_targets, test_preds)
            config.logger.info(
                f"任务 {task_id} Epoch {epoch + 1}/{config.task_epochs}, 测试损失: {test_loss:.4f}, 测试混淆矩阵:\n{test_cm}")

            writer.add_scalar("Train_Loss/Epoch", avg_loss, epoch + 1)
            writer.add_scalar("Train_Accuracy/Epoch", train_acc, epoch + 1)
            writer.add_scalar("Test_Loss", test_loss, epoch + 1)
            writer.add_scalar("Test_Accuracy", test_acc, epoch + 1)
            writer.add_scalar("Learning_Rate/Adapter", optimizer.param_groups[0]['lr'], epoch + 1)

            epoch_time = time.time() - epoch_start
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            config.logger.info(
                f"任务 {task_id} Epoch {epoch + 1}/{config.task_epochs}, 训练损失: {avg_loss:.4f}, 训练准确率: {train_acc:.2f}%, 测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%, 学习率: {optimizer.param_groups[0]['lr']}, 时间: {epoch_time:.2f}s"
            )
            scheduler.step(avg_loss)

            if config.save_frequency > 0 and (epoch + 1) % config.save_frequency == 0:
                checkpoint_file = f"checkpoints/task_{task_id}_epoch_{epoch + 1}.pth"
                torch.save(model.state_dict(), checkpoint_file)
                config.logger.info(f"保存任务 {task_id} 检查点: {checkpoint_file}")

        config.logger.info(f"任务 {task_id} 训练完成")
        return history


# 主函数
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # 初始化配置
    config = Config()
    logger = setup_logging(config.log_level)
    config.log_config(logger)

    # 创建模型
    model = ContinualLearner(config)
    config.logger.info(f"初始化模型，设备: {model.device}, 任务数: {config.num_tasks}")

    # 准备数据
    train_loaders = []
    for task_id in range(config.num_tasks):
        train_loader, _, _ = get_split_mnist(task_id, config)
        train_loaders.append(train_loader)

    # 预训练 MMoE
    pretrain_mmoe(model, train_loaders, config)

    # 验证预训练模型
    config.logger.info("验证预训练模型")
    model.eval()
    temp_heads = {
        task_id: nn.Linear(model.mmoe.experts[0][-3].out_features, 2).to(model.device)
        for task_id in range(config.num_tasks)
    }
    for _, head in temp_heads.items():
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)
    with torch.no_grad():
        for task_id in range(config.num_tasks):
            data, target = next(iter(train_loaders[task_id]))  # 使用已有 train_loaders
            data, target = data.to(model.device), target.to(model.device)
            task_id_tensor = torch.tensor([task_id], dtype=torch.long, device=model.device)
            output, _ = model.mmoe(data, task_id_tensor)
            temp_head = temp_heads[task_id]
            output = temp_head(output)
            _, predicted = torch.max(output.data, 1)
            acc = 100. * (predicted == target).sum().item() / target.size(0)
            config.logger.info(f"任务 {task_id} 预训练验证准确率: {acc:.2f}%")

    # 持续学习
    all_task_performance = []
    for task_id in range(config.num_tasks):
        config.logger.info(f"处理任务 {task_id}")
        train_loader, test_loader, classes = get_split_mnist(task_id, config)
        config.logger.info(f"任务 {task_id} 类别: {classes}")
        history = train_task(model, task_id, train_loader, test_loader, classes, config)
        all_task_performance.append(history)

        config.logger.info(f"评估所有已学习任务（任务 0 到 {task_id}）")
        task_accuracies = []
        for t in range(task_id + 1):
            _, test_loader, classes = get_split_mnist(t, config)
            model.eval()
            correct, total, test_loss = 0, 0, 0
            test_preds, test_targets = [], []
            criterion = nn.CrossEntropyLoss()
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(model.device), target.to(model.device)
                    output, _ = model(data, t)
                    loss = criterion(output, target)
                    test_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    test_preds.extend(predicted.cpu().numpy())
                    test_targets.extend(target.cpu().numpy())
            accuracy = 100. * correct / total
            test_loss = test_loss / len(test_loader)
            test_cm = confusion_matrix(test_targets, test_preds)
            task_accuracies.append(accuracy)
            config.logger.info(
                f"任务 {t} 测试损失: {test_loss:.4f}, 测试准确率: {accuracy:.2f}%, 测试混淆矩阵:\n{test_cm}")

        avg_accuracy = sum(task_accuracies) / len(task_accuracies)
        config.logger.info(f"所有任务平均准确率: {avg_accuracy:.2f}%")

    # 最终总结
    config.logger.info("训练完成！最终性能总结")
    for task_id, history in enumerate(all_task_performance):
        final_test_loss = history['test_loss'][-1]
        final_test_acc = history['test_acc'][-1]
        config.logger.info(
            f"任务 {task_id} 最终测试损失: {final_test_loss:.4f}, 最终测试准确率: {final_test_acc:.2f}%")


if __name__ == "__main__":
    main()