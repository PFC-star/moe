import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import logging
from datetime import datetime
import os

# 设置日志
def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

logger = setup_logging()

# MMoE 模型
class MMOE(nn.Module):
    def __init__(self, num_experts=4, expert_hidden=128):
        super(MMOE, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.feature_dim = 64 * 7 * 7
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, expert_hidden),
                nn.LeakyReLU(0.1),
                nn.Linear(expert_hidden, expert_hidden),
                nn.LeakyReLU(0.1)
            ) for _ in range(num_experts)
        ])
        self.task_embedding = nn.Embedding(10, 16)
        self.gate = nn.Sequential(
            nn.Linear(self.feature_dim + 16, num_experts),
            nn.Softmax(dim=-1)
        )
        self.gate = nn.Sequential(
            nn.Linear(self.feature_dim + 16, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, task_id):
        features = self.feature_extractor(x)
        task_embed = self.task_embedding(task_id)
        task_embed = task_embed.expand(features.size(0), -1)
        gate_input = torch.cat([features, task_embed], dim=1)
        gate_weights = self.gate(gate_input)
        expert_outputs = [expert(features) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        weighted_output = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)
        return weighted_output

# 适配器模块
class Adapter(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(Adapter, self).__init__()
        bottleneck_dim = 64  # 增加瓶颈维度
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 持续学习模型
class ContinualLearner(nn.Module):
    def __init__(self, num_experts=4, expert_hidden=128):
        super(ContinualLearner, self).__init__()
        self.mmoe = MMOE(num_experts, expert_hidden)
        self.adapters = nn.ModuleDict()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def add_task(self, task_id, num_classes=2):
        expert = self.mmoe.experts[0]
        for layer in reversed(expert):
            if isinstance(layer, nn.Linear):
                in_dim = layer.out_features
                break
        else:
            raise ValueError("专家网络中未找到 nn.Linear 层")
        adapter = Adapter(in_dim, num_classes).to(self.device)
        self.adapters[str(task_id)] = adapter
        logger.info(f"任务 {task_id} 添加适配器，输入维度: {in_dim}, 输出维度: {num_classes}")

    def forward(self, x, task_id):
        x = x.to(self.device)
        task_id_tensor = torch.tensor([task_id], dtype=torch.long, device=self.device)
        shared_output = self.mmoe(x, task_id_tensor)
        adapter_key = str(task_id)
        if adapter_key in self.adapters:
            output = self.adapters[adapter_key](shared_output)
        else:
            raise ValueError(f"未找到任务 {task_id} 的适配器")
        return output

# 数据集准备
def get_split_mnist(task_id, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
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
    train_loader = DataLoader(remapped_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(remapped_test_dataset, batch_size=batch_size, shuffle=False)

    # 验证标签分布
    train_labels = np.array([target for _, target in remapped_train_dataset])
    test_labels = np.array([target for _, target in remapped_test_dataset])
    logger.debug(f"任务 {task_id} 训练标签分布: {np.bincount(train_labels)}")
    logger.debug(f"任务 {task_id} 测试标签分布: {np.bincount(test_labels)}")

    data, _ = next(iter(train_loader))
    logger.debug(f"任务 {task_id} 输入数据形状: {data.shape}, 值范围: [{data.min().item()}, {data.max().item()}]")

    return train_loader, test_loader, classes

# 预训练 MMoE
def pretrain_mmoe(model, train_loaders, num_epochs=5, checkpoint_path="checkpoints/mmoe_weights.pth",pretrained=False):
    writer = SummaryWriter(log_dir=f"runs/pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if os.path.exists(checkpoint_path) and pretrained:
        try:
            logger.info(f"找到预训练权重文件 {checkpoint_path}，尝试加载")
            model.mmoe.load_state_dict(torch.load(checkpoint_path, map_location=model.device))
            logger.info(f"成功加载预训练权重 {checkpoint_path}")
            writer.close()
            return
        except Exception as e:
            logger.error(f"加载预训练权重失败: {e}，将重新预训练")

    optimizer = optim.Adam(model.mmoe.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"开始预训练 MMoE，任务数: {len(train_loaders)}, Epochs: {num_epochs}, 学习率: {optimizer.param_groups[0]['lr']}")

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        epoch_start = time.time()

        for task_id, train_loader in enumerate(train_loaders):
            temp_head = nn.Linear(model.mmoe.experts[0][-2].out_features, 2).to(model.device)
            temp_optimizer = optim.Adam(list(model.mmoe.parameters()) + list(temp_head.parameters()), lr=0.001)
            labels = np.concatenate([target.cpu().numpy() for _, target in train_loader])
            logger.debug(f"任务 {task_id} 训练标签分布: {np.bincount(labels)}")

            for data, target in train_loader:
                data, target = data.to(model.device), target.to(model.device)
                task_id_tensor = torch.tensor([task_id], dtype=torch.long, device=model.device)

                temp_optimizer.zero_grad()
                output = model.mmoe(data, task_id_tensor)
                output = temp_head(output)
                loss = criterion(output, target)

                loss.backward()
                grad_norm = sum(p.grad.norm().item() for p in model.mmoe.parameters() if p.grad is not None)
                logger.debug(f"Epoch {epoch + 1}, 任务 {task_id}, Batch 梯度范数: {grad_norm:.4f}")
                torch.nn.utils.clip_grad_norm_(list(model.mmoe.parameters()) + list(temp_head.parameters()), max_norm=1.0)
                temp_optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = total_loss / sum(len(loader) for loader in train_loaders)
        train_acc = 100. * correct / total

        # 测试
        model.eval()
        test_accs = []
        for task_id in range(len(train_loaders)):
            _, test_loader, classes = get_split_mnist(task_id)
            test_correct, test_total = 0, 0
            temp_head = nn.Linear(model.mmoe.experts[0][-2].out_features, 2).to(model.device)
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(model.device), target.to(model.device)
                    task_id_tensor = torch.tensor([task_id], dtype=torch.long, device=model.device)
                    output = model.mmoe(data, task_id_tensor)
                    output = temp_head(output)
                    _, predicted = torch.max(output.data, 1)
                    test_total += target.size(0)
                    test_correct += (predicted == target).sum().item()
            test_acc = 100. * test_correct / test_total
            test_accs.append(test_acc)
            logger.info(f"预训练 Epoch {epoch + 1}/{num_epochs}, 任务 {task_id} 测试准确率: {test_acc:.2f}%")
            writer.add_scalar(f"Test_Accuracy/Task_{task_id}", test_acc, epoch + 1)

        avg_test_acc = sum(test_accs) / len(test_accs)
        writer.add_scalar("Train_Loss", avg_loss, epoch + 1)
        writer.add_scalar("Train_Accuracy", train_acc, epoch + 1)
        writer.add_scalar("Test_Accuracy/Average", avg_test_acc, epoch + 1)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch + 1)

        epoch_time = time.time() - epoch_start
        logger.info(
            f"预训练 Epoch {epoch + 1}/{num_epochs}, 训练损失: {avg_loss:.4f}, 训练准确率: {train_acc:.2f}%, 平均测试准确率: {avg_test_acc:.2f}%, 时间: {epoch_time:.2f}s"
        )
        scheduler.step(avg_loss)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.mmoe.state_dict(), checkpoint_path)
    logger.info(f"MMoE 预训练完成，权重已保存至 {checkpoint_path}")
    writer.close()

# 训练新任务
def train_task(model, task_id, train_loader, test_loader, classes, num_epochs=5):
    writer = SummaryWriter(log_dir=f"runs/task_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    model.add_task(task_id, num_classes=len(classes))
    optimizer = optim.Adam(model.adapters.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    logger.info(
        f"开始训练任务 {task_id}，类别: {classes}, Epochs: {num_epochs}, 训练样本: {len(train_loader.dataset)}, 测试样本: {len(test_loader.dataset)}, 学习率: {optimizer.param_groups[0]['lr']}"
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        epoch_start = time.time()
        pbar = tqdm(train_loader, desc=f'Task {task_id} Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(model.device), target.to(model.device)
            if batch_idx == 0:
                logger.debug(f"任务 {task_id}, Epoch {epoch + 1}, Batch 标签样本: {target[:5].cpu().numpy()}")

            optimizer.zero_grad()
            output = model(data, task_id)
            loss = criterion(output, target)
            loss.backward()

            grad_norm = sum(p.grad.norm().item() for p in model.adapters.parameters() if p.grad is not None)
            logger.debug(f"任务 {task_id}, Epoch {epoch + 1}, Batch {batch_idx + 1}, 梯度范数: {grad_norm:.4f}")
            torch.nn.utils.clip_grad_norm_(model.adapters.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100. * correct / total:.2f}%'})

        avg_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(model.device), target.to(model.device)
                output = model(data, task_id)
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        test_acc = 100. * test_correct / test_total

        writer.add_scalar("Train_Loss", avg_loss, epoch + 1)
        writer.add_scalar("Train_Accuracy", train_acc, epoch + 1)
        writer.add_scalar("Test_Accuracy", test_acc, epoch + 1)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch + 1)

        epoch_time = time.time() - epoch_start
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        logger.info(
            f"任务 {task_id} Epoch {epoch + 1}/{num_epochs}, 训练损失: {avg_loss:.4f}, 训练准确率: {train_acc:.2f}%, 测试准确率: {test_acc:.2f}%, 时间: {epoch_time:.2f}s"
        )
        scheduler.step(avg_loss)

    logger.info(f"任务 {task_id} 训练完成")
    writer.close()
    return history

# 主函数
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    model = ContinualLearner(num_experts=4, expert_hidden=128)
    num_tasks = 5

    logger.info(f"初始化模型，设备: {model.device}, 任务数: {num_tasks}")
    train_loaders = []
    for task_id in range(num_tasks):
        train_loader, _, _ = get_split_mnist(task_id)
        train_loaders.append(train_loader)

    pretrain_mmoe(model, train_loaders, num_epochs=5, checkpoint_path="checkpoints/mmoe_weights.pth",pretrained=False)

    # 验证预训练模型
    logger.info("验证预训练模型")
    model.eval()
    with torch.no_grad():
        for task_id in range(num_tasks):
            train_loader, _, _ = get_split_mnist(task_id)
            data, target = next(iter(train_loader))
            data, target = data.to(model.device), target.to(model.device)
            task_id_tensor = torch.tensor([task_id], dtype=torch.long, device=model.device)
            output = model.mmoe(data, task_id_tensor)
            temp_head = nn.Linear(model.mmoe.experts[0][-2].out_features, 2).to(model.device)
            output = temp_head(output)
            _, predicted = torch.max(output.data, 1)
            acc = 100. * (predicted == target).sum().item() / target.size(0)
            logger.info(f"任务 {task_id} 预训练验证准确率: {acc:.2f}%")

    all_task_performance = []
    for task_id in range(num_tasks):
        logger.info(f"处理任务 {task_id}")
        train_loader, test_loader, classes = get_split_mnist(task_id)
        logger.info(f"任务 {task_id} 类别: {classes}")
        history = train_task(model, task_id, train_loader, test_loader, classes)
        all_task_performance.append(history)

        logger.info(f"评估所有已学习任务（任务 0 到 {task_id}）")
        task_accuracies = []
        for t in range(task_id + 1):
            _, test_loader, classes = get_split_mnist(t)
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(model.device), target.to(model.device)
                    output = model(data, t)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            accuracy = 100. * correct / total
            task_accuracies.append(accuracy)
            logger.info(f"任务 {t} 测试准确率: {accuracy:.2f}%")

        avg_accuracy = sum(task_accuracies) / len(task_accuracies)
        logger.info(f"所有任务平均准确率: {avg_accuracy:.2f}%")

    logger.info("训练完成！最终性能总结")
    for task_id, history in enumerate(all_task_performance):
        final_test_acc = history['test_acc'][-1]
        logger.info(f"任务 {task_id} 最终测试准确率: {final_test_acc:.2f}%")

if __name__ == "__main__":
    main()