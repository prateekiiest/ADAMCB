import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
from torch.func import functional_call, vmap, grad

# Ensure pandas displays all columns nicely in the console
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Wrapper to strictly track dataset Index IDs
class IndexedSubset(Dataset):
    def __init__(self, subset):
        self.subset = subset
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        data, target = self.subset[idx]
        return idx, data, target

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        # Modified for CIFAR-10: 3 channels * 32 * 32 = 3072 input features
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class adam_optimizer():
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.model = model
        self.m = {name: torch.zeros_like(p.data) for name, p in self.model.named_parameters()}
        self.v = {name: torch.zeros_like(p.data) for name, p in self.model.named_parameters()}
        self.t = 0

    def zero_grad(self):
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def step(self):
        self.t += 1
        beta1, beta2 = self.betas
        for name, param in self.model.named_parameters():
            if param.grad is None: continue
            g = param.grad
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * g
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * (g ** 2)
            m_hat = self.m[name] / (1 - beta1 ** self.t)
            v_hat = self.v[name] / (1 - beta2 ** self.t)
            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

class ParallelDependentRounding:
    def __init__(self, probabilities_tensor, k, device='cpu'):
        self.device = device
        self.probs = probabilities_tensor.clone().to(self.device, dtype=torch.float64)
        self.k = k
        # Modfied to process 32 pairs = 64 datapoints in parallel
        self.k_temp = 256 
        self.epsilon = 1e-9

    def round(self):
        while True:
            floating_mask = (self.probs > self.epsilon) & (self.probs < (1.0 - self.epsilon))
            floating_indices = torch.nonzero(floating_mask).squeeze(1)
            n_floating = len(floating_indices)
            
            if n_floating < 2:
                break
                
            shuffle_idx = torch.randperm(n_floating, device=self.device)
            shuffled_floating_indices = floating_indices[shuffle_idx]
            
            n_pairs = min(self.k_temp, n_floating // 2)
            idx1 = shuffled_floating_indices[:n_pairs]
            idx2 = shuffled_floating_indices[n_pairs:2*n_pairs]
            
            val1 = self.probs[idx1]
            val2 = self.probs[idx2]
            
            alpha = torch.minimum(1.0 - val1, val2)
            beta = torch.minimum(val1, 1.0 - val2)
            
            prob_scenario_1 = beta / (alpha + beta)
            
            rand_vals = torch.rand(n_pairs, device=self.device, dtype=torch.float64)
            scenario_1_mask = rand_vals < prob_scenario_1
            
            update_val1 = torch.where(scenario_1_mask, alpha, -beta)
            update_val2 = torch.where(scenario_1_mask, -alpha, beta)
            
            self.probs[idx1] += update_val1
            self.probs[idx2] += update_val2
            
            self.probs = torch.clamp(self.probs, 0.0, 1.0)
            self.probs = torch.where(self.probs < self.epsilon, torch.zeros_like(self.probs), self.probs)
            self.probs = torch.where(self.probs > 1.0 - self.epsilon, torch.ones_like(self.probs), self.probs)

        _, top_k_indices = torch.topk(self.probs, self.k)
        return top_k_indices.cpu().numpy()
        
class ADAMCB:
    def __init__(self, model, train_dataset, K, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, gamma=0.1, device='cpu'):
        self.model = model
        self.train_dataset = train_dataset
        self.K = K
        self.n = len(train_dataset)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.gamma = gamma
        self.device = device

        self.m = {name: torch.zeros_like(p.data) for name, p in self.model.named_parameters()}
        self.v = {name: torch.zeros_like(p.data) for name, p in self.model.named_parameters()}
        self.t = 0
        
        self.w_t = np.ones(self.n)
        self.p_t = self.K * (np.ones(self.n) / self.n)
        self.indices = np.zeros(K, dtype=int)
        self.G_t_hat = []
        self.gradient_norms = []
        self.loss_function = nn.CrossEntropyLoss()
        
        self.seen_indices = set()

    def getAlpha(self, temp, w_sorted):
        sum_weight = np.sum(w_sorted)
        for i in range(len(w_sorted)):
            alpha = (temp * sum_weight) / (1.0 - i * temp)
            curr = w_sorted[i]
            if alpha > curr:
                return alpha
            sum_weight = sum_weight - curr
        return 1.0

    def batch_selection_adamcb(self):
        w_temp = self.w_t.copy()
        theSum = np.sum(w_temp)
        temp = (1.0 / self.K - self.gamma / self.n) / (1.0 - self.gamma)
        
        if np.max(w_temp) >= temp * theSum:
            w_sorted = np.sort(w_temp)[::-1]
            alpha_t = self.getAlpha(temp, w_sorted)
            S_null = np.nonzero(w_temp >= alpha_t)[0]
            for s in S_null:
                w_temp[s] = alpha_t
        else:
            S_null = []
            
        weights_sum = np.sum(w_temp)
        self.p_t = self.K * ((1.0 - self.gamma) * (w_temp / weights_sum) + (self.gamma / self.n))
        
        p_t_tensor = torch.tensor(self.p_t, device=self.device, dtype=torch.float32)
        dr = ParallelDependentRounding(p_t_tensor, self.K, device=self.device)
        self.indices = dr.round()
        
        self.seen_indices.update(self.indices.tolist())
        return S_null

    def compute_unbiased_gradient_estimate(self):
        batch_items = [self.train_dataset[k] for k in self.indices]
        data_batch = torch.stack([item[1] for item in batch_items]).to(self.device, non_blocking=True)
        target_batch = torch.tensor([item[2] for item in batch_items], device=self.device, dtype=torch.long)
    
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())

        def compute_loss(params, buffers, sample, target):
            sample = sample.unsqueeze(0)
            target = target.unsqueeze(0)
            predictions = functional_call(self.model, (params, buffers), (sample,))
            return self.loss_function(predictions, target)

        ft_compute_grad = grad(compute_loss)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        sample_grads = ft_compute_sample_grad(params, buffers, data_batch, target_batch)

        self.G_t_hat = []
        per_sample_norm = torch.zeros(self.K, device=self.device)
        for name, param in self.model.named_parameters():
            per_sample_grad = sample_grads[name]
            per_sample_norm += (per_sample_grad.flatten(start_dim=1) ** 2).sum(dim=1)
    
        self.gradient_norms = torch.sqrt(per_sample_norm) 
    
        p_t_tensor = torch.tensor(self.p_t, device=self.device, dtype=torch.float32)
        indices_tensor = torch.tensor(self.indices, device=self.device, dtype=torch.long)
        scaling_factor = 1.0 / (self.n * p_t_tensor[indices_tensor])

        for name, param in self.model.named_parameters():
            per_sample_grad = sample_grads[name]
            view_shape = [-1] + [1] * (per_sample_grad.dim() - 1)
            scaled_grad = per_sample_grad * scaling_factor.view(*view_shape)
            self.G_t_hat.append(scaled_grad.sum(dim=0) / self.K)
    
        with torch.no_grad():
            output = self.model(data_batch)
            loss = self.loss_function(output, target_batch)
    
        return loss

    def update_model_parameters(self):
        self.t += 1
        beta1, beta2 = self.betas
        bias_correction1 = 1 - beta1 ** self.t
        bias_correction2 = 1 - beta2 ** self.t
        
        for i, (name, param) in enumerate(self.model.named_parameters()):
            g = self.G_t_hat[i]
            self.m[name].mul_(beta1).add_(g, alpha=1 - beta1)
            self.v[name].mul_(beta2).addcmul_(g, g, value=1 - beta2)
            
            m_hat = self.m[name] / bias_correction1
            v_hat = self.v[name] / bias_correction2
            param.data.addcdiv_(m_hat, torch.sqrt(v_hat).add_(self.eps), value=-self.lr)

    def update_sample_weights(self, S_null):
        with torch.no_grad():
            w_temp = torch.tensor(self.w_t, device=self.device, dtype=torch.float64)
            p_t_tensor = torch.tensor(self.p_t, device=self.device, dtype=torch.float64)
            indices_tensor = torch.tensor(self.indices, device=self.device, dtype=torch.long)
            
            grad_norms = self.gradient_norms.to(torch.float64)
            selected_p_t = p_t_tensor[indices_tensor]
            
            p_min = self.gamma / self.n
            L = torch.max(grad_norms) + 1e-8
            
            loss_val = -(grad_norms) / (selected_p_t ) + (L) / (p_min)
            
            h_hat_updates = loss_val / selected_p_t
            mu_t = torch.mean(h_hat_updates)
            h_hat_stable = h_hat_updates - mu_t
            alpha_p = (self.gamma / self.n) * np.sqrt(self.K / (self.n * self.t))
            h_hat_full = torch.zeros(self.n, device=self.device, dtype=torch.float64)
            h_hat_full.scatter_(0, indices_tensor, h_hat_stable)
            new_w_t = w_temp * torch.exp(-alpha_p * h_hat_full).clamp(0.01, 10.0)
            
            if len(S_null) > 0:
                s_null_tensor = torch.tensor(S_null, device=self.device, dtype=torch.long)
                new_w_t[s_null_tensor] = w_temp[s_null_tensor]
                
            self.w_t = new_w_t.detach().cpu().numpy()
            
            floor_count = np.sum(self.w_t <= 1e-4)
            ceil_count = np.sum(self.w_t >= 0.99)
            return floor_count, ceil_count, grad_norms.min().item(), grad_norms.max().item(), grad_norms.mean().item()
        
    def step(self):
        prev_w = self.w_t.copy()
        S_null = self.batch_selection_adamcb()
        prev_p = self.p_t.copy() 
        
        loss = self.compute_unbiased_gradient_estimate()
        self.update_model_parameters()
        raw_grads = self.gradient_norms.detach().cpu().numpy()
        
        floor_c, ceil_c, min_g, max_g, mean_g = self.update_sample_weights(S_null)
        
        new_w = self.w_t.copy()
        new_w_sum = np.sum(new_w)
        new_p = self.K * ((1.0 - self.gamma) * (new_w / new_w_sum) + (self.gamma / self.n))
        
        return loss, floor_c, ceil_c, min_g, max_g, mean_g, prev_w, prev_p, new_w, new_p, raw_grads, self.indices
    
def fast_evaluate(model, data_loader, device, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                _, data, target = batch
            else:
                data, target = batch
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100. * correct / len(data_loader.dataset)
    loss = total_loss / len(data_loader.dataset)
    return loss, acc

if __name__ == '__main__':
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard CIFAR-10 Normalization
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Combine CIFAR-10 Train and Test sets to form the full dataset
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    full_cifar_dataset = ConcatDataset([cifar_train, cifar_test])
    
    total_len = len(full_cifar_dataset)
    train_len = int(0.9 * total_len)
    test_len = total_len - train_len
    
    # 90% Train, 10% Test split across the whole CIFAR-10 dataset
    train_subset, test_subset = torch.utils.data.random_split(full_cifar_dataset, [train_len, test_len])
    
    train_indexed = IndexedSubset(train_subset)
    test_indexed = IndexedSubset(test_subset)
    
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    BETAS = (0.9, 0.999)
    EPSILON = 1e-8
    EPOCHS = 10
    GAMMA = 0.1
    
    steps_per_epoch = len(train_indexed) // BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch

    adam_loader = DataLoader(train_indexed, batch_size=BATCH_SIZE, shuffle=True)
    adam_iter = iter(adam_loader)
    
    test_eval_loader = DataLoader(test_indexed, batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    model_adam = MLP().to(device)
    init_state = {k: v.clone() for k, v in model_adam.state_dict().items()}
    
    model_adamcb = MLP().to(device)
    model_adamcb.load_state_dict(init_state)

    optimizer_adam = adam_optimizer(model_adam, lr=LEARNING_RATE, betas=BETAS, eps=EPSILON)
    adamcb_optimizer = ADAMCB(model_adamcb, train_indexed, K=BATCH_SIZE, lr=LEARNING_RATE, betas=BETAS, eps=EPSILON, gamma=GAMMA, device=device)
    
    adam_seen_indices = set()
    
    adam_train_losses, adam_test_losses, adam_test_accs = [], [], []
    adamcb_train_losses, adamcb_test_losses, adamcb_test_accs = [], [], []
    
    print(f"\nStarting CIFAR-10 Experiment: {total_steps} total batches ({train_len} Train, {test_len} Test)...")
    
    for step in range(1, total_steps + 1):
        model_adam.train()
        try:
            idx_adam, data_adam, target_adam = next(adam_iter)
        except StopIteration:
            adam_iter = iter(adam_loader)
            idx_adam, data_adam, target_adam = next(adam_iter)
            
        adam_seen_indices.update(idx_adam.tolist())
        data_adam, target_adam = data_adam.to(device), target_adam.to(device)
        
        params_adam = dict(model_adam.named_parameters())
        buffers_adam = dict(model_adam.named_buffers())

        def compute_loss_adam(params, buffers, sample, target):
            predictions = functional_call(model_adam, (params, buffers), (sample.unsqueeze(0),))
            return criterion(predictions, target.unsqueeze(0))

        sample_grads_adam = vmap(grad(compute_loss_adam), in_dims=(None, None, 0, 0))(params_adam, buffers_adam, data_adam, target_adam)
        
        adam_squared_sums = torch.zeros(BATCH_SIZE, device=device)
        for name, param in model_adam.named_parameters():
            adam_squared_sums += (sample_grads_adam[name].flatten(start_dim=1) ** 2).sum(dim=1)
        adam_raw_grads = torch.sqrt(adam_squared_sums).detach().cpu().numpy()
        
        optimizer_adam.zero_grad()
        output = model_adam(data_adam)
        adam_loss = criterion(output, target_adam)
        adam_loss.backward()
        optimizer_adam.step()
        
        model_adamcb.train()
        adamcb_loss, floor_c, ceil_c, min_g, max_g, mean_g, prev_w, prev_p, new_w, new_p, cb_raw_grads, sampled_indices = adamcb_optimizer.step()
        
        adam_test_loss, adam_test_acc = fast_evaluate(model_adam, test_eval_loader, device, criterion)
        adamcb_test_loss, adamcb_test_acc = fast_evaluate(model_adamcb, test_eval_loader, device, criterion)
        
        adam_train_losses.append(adam_loss.item())
        adam_test_losses.append(adam_test_loss)
        adam_test_accs.append(adam_test_acc)
        
        adamcb_train_losses.append(adamcb_loss.item())
        adamcb_test_losses.append(adamcb_test_loss)
        adamcb_test_accs.append(adamcb_test_acc)
        
        print(f"\n[{'='*30} STEP {step}/{total_steps} (Epoch {(step-1)//steps_per_epoch + 1}) {'='*30}]")
        print("[PERFORMANCE COMPARISON]")
        print(f"ADAM   | Train Loss: {adam_loss.item():.4f} | Test Loss: {adam_test_loss:.4f} | Test Acc: {adam_test_acc:.2f}%")
        print(f"ADAMCB | Train Loss: {adamcb_loss.item():.4f} | Test Loss: {adamcb_test_loss:.4f} | Test Acc: {adamcb_test_acc:.2f}%")
        print("-" * 80)
        print("[DIAGNOSTICS & BEHAVIOR]")
        print(f"ADAM   -> Unique Items Seen: {len(adam_seen_indices)} / {train_len}")
        print(f"ADAMCB -> Unique Items Seen: {len(adamcb_optimizer.seen_indices)} / {train_len}")
        print(f"ADAMCB -> Dataset Weights  : {floor_c} easy items (floor 0.1) | {ceil_c} hard items (ceil 1.0)")
        print(f"ADAM   -> Batch Gradients  : Min={adam_raw_grads.min():.4f} | Mean={adam_raw_grads.mean():.4f} | Max={adam_raw_grads.max():.4f}")
        print(f"ADAMCB -> Batch Gradients  : Min={min_g:.4f} | Mean={mean_g:.4f} | Max={max_g:.4f}")

    print("\nGenerating and saving comparison plots...")
    window_size = 5 
    def smooth_data(data, window):
        return [np.mean(data[i:i+window]) for i in range(0, len(data), window)]

    adam_train_smooth = smooth_data(adam_train_losses, window_size)
    adamcb_train_smooth = smooth_data(adamcb_train_losses, window_size)
    adam_test_smooth = smooth_data(adam_test_losses, window_size)
    adamcb_test_smooth = smooth_data(adamcb_test_losses, window_size)
    adam_acc_smooth = smooth_data(adam_test_accs, window_size)
    adamcb_acc_smooth = smooth_data(adamcb_test_accs, window_size)
    
    x_axis = [min(i + window_size, len(adam_train_losses)) for i in range(0, len(adam_train_losses), window_size)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, adam_train_smooth, label='ADAM', alpha=0.8, linewidth=2, color='blue')
    plt.plot(x_axis, adamcb_train_smooth, label='ADAMCB', alpha=0.8, linewidth=2, color='orange')
    plt.xlabel('Step (Batch)')
    plt.ylabel(f'Train Loss (Avg over {window_size} steps)')
    plt.title('Train Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_loss_comparison.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, adam_test_smooth, label='ADAM', alpha=0.8, linewidth=2, color='blue')
    plt.plot(x_axis, adamcb_test_smooth, label='ADAMCB', alpha=0.8, linewidth=2, color='orange')
    plt.xlabel('Step (Batch)')
    plt.ylabel(f'Test Loss (Avg over {window_size} steps)')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('test_loss_comparison.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, adam_acc_smooth, label='ADAM', alpha=0.8, linewidth=2, color='blue')
    plt.plot(x_axis, adamcb_acc_smooth, label='ADAMCB', alpha=0.8, linewidth=2, color='orange')
    plt.xlabel('Step (Batch)')
    plt.ylabel(f'Test Accuracy (%) (Avg over {window_size} steps)')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('test_accuracy_comparison.png')
    plt.close()
    
    print("Training complete! Plots saved.")