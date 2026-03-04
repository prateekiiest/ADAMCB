import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
from torch.func import functional_call, vmap, grad

# Ensure pandas displays all columns nicely in the console
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
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
    def __init__(self, probabilities_tensor, k, device='cuda'):
        self.device = device
        self.probs = probabilities_tensor.clone().to(self.device, dtype=torch.float64)
        self.k = k
        self.k_temp = 32
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
    def __init__(self, model, train_dataset, K, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, gamma=0.2, device='cpu'):
        self.model = model
        self.train_dataset = train_dataset
        self.K = K
        self.n = len(train_dataset)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.gamma = gamma
        self.device = device

        self.p_min = self.gamma / self.n
        self.alpha_p = float(self.p_min**3)

        self.m = {name: torch.zeros_like(p.data) for name, p in self.model.named_parameters()}
        self.v = {name: torch.zeros_like(p.data) for name, p in self.model.named_parameters()}
        self.t = 0
        
        # Training time variables saved as class variables
        self.w_t = np.ones(self.n)
        self.p_t = np.ones(self.n) / self.n
        self.indices = np.zeros(K, dtype=int)
        self.G_t_hat = []
        self.gradient_norms = []
        
        # self.max_norm = float('-inf')
        # self.min_norm = float('inf')
        self.max_norm = float('-inf')
        self.min_norm = float('inf')
        self.L_max = 0.0 # NEW: Track the empirical L constant
        self.loss_function = nn.CrossEntropyLoss()

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
        theSum = np.sum(self.w_t)
        temp = (1.0 / self.K - self.gamma / self.n) / (1.0 - self.gamma)
        w_temp = self.w_t.copy()
        
        if np.max(self.w_t) >= temp * theSum:
            w_sorted = np.sort(self.w_t)[::-1]
            alpha_t = self.getAlpha(temp, w_sorted)
            S_null = np.nonzero(w_temp >= alpha_t)[0]
            for s in S_null:
                w_temp[s] = alpha_t
        else:
            S_null = []
            
        weights_sum = np.sum(w_temp)
        # Calculate p_t
        self.p_t = self.K * ((1.0 - self.gamma) * (w_temp / weights_sum) + (self.gamma / self.n))
        
        # 1. Convert to tensor and move to GPU
        p_t_tensor = torch.tensor(self.p_t, device=self.device, dtype=torch.float32)
        
        # 2. Run the insanely fast parallel rounding
        dr = ParallelDependentRounding(p_t_tensor, self.K, device=self.device)
        
        # 3. It returns the exact indices directly
        self.indices = dr.round()
        print(len(self.indices), "indices selected in this batch.")
        
        return S_null

    def compute_unbiased_gradient_estimate(self):
        # 1. Fast Data Extraction (Removed tqdm blocking, optimized list ops)
        # Assuming train_dataset returns (tensor_image, int_label)
        batch_items = [self.train_dataset[k] for k in self.indices]
        data_batch = torch.stack([item[0] for item in batch_items]).to(self.device, non_blocking=True)
        target_batch = torch.tensor([item[1] for item in batch_items], device=self.device, dtype=torch.long)
        
        # 2. Extract model state for functional calls
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())

        def compute_loss(params, buffers, sample, target):
            sample = sample.unsqueeze(0)
            target = target.unsqueeze(0)
            predictions = functional_call(self.model, (params, buffers), (sample,))
            return self.loss_function(predictions, target)

        # 3. Natively parallelized gradient calculation via C++/CUDA
        ft_compute_grad = grad(compute_loss)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        sample_grads = ft_compute_sample_grad(params, buffers, data_batch, target_batch)

        self.G_t_hat = []
        squared_sums = torch.zeros(self.K, device=self.device)
        
        # 4. Fully GPU-Vectorized Probability Math (No NumPy or list comprehensions)
        p_t_tensor = torch.tensor(self.p_t, device=self.device, dtype=torch.float32)
        indices_tensor = torch.tensor(self.indices, device=self.device, dtype=torch.long)
        
        # Extract batch probabilities and normalize natively on the GPU
        p_t_batch = p_t_tensor[indices_tensor] / p_t_tensor.sum()
        
        # Pre-calculate the scaling factor to avoid repeating divisions
        scaling_factor = 1.0 / (self.n * p_t_batch)  # Shape: (K,)

        # 5. Process gradients efficiently
        for name, param in self.model.named_parameters():
            per_sample_grad = sample_grads[name] 
            
            # Use flatten() instead of view() to avoid contiguous memory errors, sum squares
            squared_sums += (per_sample_grad.flatten(start_dim=1) ** 2).sum(dim=1)
            
            # Reshape scaling_factor to broadcast across all parameter dimensions: (K, 1, 1, ...)
            view_shape = [-1] + [1] * (per_sample_grad.dim() - 1)
            
            # Multiply instead of divide (multiplication is faster on GPUs)
            scaled_grad = per_sample_grad * scaling_factor.view(*view_shape)
            
            # Average and append
            self.G_t_hat.append(scaled_grad.sum(dim=0) / self.K)
            
        # 6. Keep the norms strictly on the GPU (Removed .tolist())
        self.gradient_norms = torch.sqrt(squared_sums)
        
        # 7. Fast forward pass for the batch loss
        with torch.no_grad():
            output = self.model(data_batch)
            loss = self.loss_function(output, target_batch)
            
        return loss

    def update_model_parameters(self):
        self.t += 1
        beta1, beta2 = self.betas
        
        # Precompute bias corrections
        bias_correction1 = 1 - beta1 ** self.t
        bias_correction2 = 1 - beta2 ** self.t
        
        for i, (name, param) in enumerate(self.model.named_parameters()):
            g = self.G_t_hat[i]
            
            # Equivalent to: self.m[name] = beta1 * self.m[name] + (1 - beta1) * g
            # But runs strictly in-place on the GPU
            self.m[name].mul_(beta1).add_(g, alpha=1 - beta1)
            
            # Equivalent to: self.v[name] = beta2 * self.v[name] + (1 - beta2) * (g ** 2)
            self.v[name].mul_(beta2).addcmul_(g, g, value=1 - beta2)
            
            # We don't want to modify the core m and v states, so we make temporary hat tensors
            m_hat = self.m[name] / bias_correction1
            v_hat = self.v[name] / bias_correction2
            
            # Equivalent to: param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            # Uses addcdiv_ for highly optimized fused addition/division on the GPU
            param.data.addcdiv_(m_hat, torch.sqrt(v_hat).add_(self.eps), value=-self.lr)

    def update_sample_weights(self, S_null):
        with torch.no_grad():
            w_temp = torch.tensor(self.w_t, device=self.device, dtype=torch.float32)
            p_t_tensor = torch.tensor(self.p_t, device=self.device, dtype=torch.float32)
            indices_tensor = torch.tensor(self.indices, device=self.device, dtype=torch.long)
            grad_norms_tensor = self.gradient_norms

            # 1. Dynamically update L (the max gradient norm observed so far)
            current_max_norm = torch.max(grad_norms_tensor).item()
            if current_max_norm > self.L_max:
                self.L_max = current_max_norm

            h_hat = torch.zeros(self.n, device=self.device, dtype=torch.float32)
            normalized_p_t = p_t_tensor / torch.sum(p_t_tensor)
            selected_p_t = normalized_p_t[indices_tensor]
            
            # 2. EXACT FORMULA REPLICATION
            # Part A: -||grad||^2 / p_i^2
            part_a = -(grad_norms_tensor ** 2) / (selected_p_t ** 2)
            
            # Part B: + L^2 / p_min^2
            part_b = (self.L_max ** 2) / (self.p_min ** 2)
            
            # Combine and divide by (K * p_i)
            # loss_val matches the exact mathematical definition of \hat{\ell}_i(t)
            loss_val = (part_a + part_b) / (self.K * selected_p_t)
            
            # Scatter back to the full N-sized tensor
            h_hat.scatter_(0, indices_tensor, loss_val)
            
            # 3. Apply the exponent update: w_{t+1} = w_t * exp(-alpha * h_hat)
            new_w_t = w_temp * torch.exp(-self.alpha_p * h_hat)
            
            # 4. Handle S_null (clipping bounds)
            if len(S_null) > 0:
                s_null_tensor = torch.tensor(S_null, device=self.device, dtype=torch.long)
                new_w_t[s_null_tensor] = w_temp[s_null_tensor]
                
            self.w_t = new_w_t.detach().cpu().numpy()
            
            
            
    def step(self):
        S_null = self.batch_selection_adamcb()
        loss = self.compute_unbiased_gradient_estimate()
        
        # WE REMOVED THE ARTIFICIAL NORMALIZATION HERE 
        # The raw self.gradient_norms are passed directly to the update functions.
            
        self.update_model_parameters()
        self.update_sample_weights(S_null)
        
        return loss


def fast_evaluate(model, data_loader, device, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100. * correct / len(data_loader.dataset)
    loss = total_loss / len(data_loader.dataset)
    return loss, acc

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    DATASET_SIZE = 50000 
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    BETAS = (0.9, 0.999)
    EPSILON = 1e-8
    EPOCHS = 5
    GAMMA = 0.1

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    full_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_subset = Subset(full_train, range(DATASET_SIZE))
    test_subset = Subset(full_test, range(1000)) 

    adam_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    adam_iter = iter(adam_loader)
    
    test_eval_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    model_adam = MLP().to(device)
    init_state = {k: v.clone() for k, v in model_adam.state_dict().items()}
    
    model_adamcb = MLP().to(device)
    model_adamcb.load_state_dict(init_state)

    optimizer_adam = adam_optimizer(model_adam, lr=LEARNING_RATE, betas=BETAS, eps=EPSILON)
    adamcb_optimizer = ADAMCB(model_adamcb, train_subset, K=BATCH_SIZE, lr=LEARNING_RATE, betas=BETAS, eps=EPSILON, gamma=GAMMA, device=device)
    
    steps_per_epoch = DATASET_SIZE // BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch
    
    # Store trajectory variables for plotting
    adam_train_losses = []
    adam_test_losses = []
    adam_test_accs = []
    
    adamcb_train_losses = []
    adamcb_test_losses = []
    adamcb_test_accs = []
    
    print(f"\nStarting Parallel Training: {total_steps} total batches...")
    
    for step in range(1, total_steps + 1):
        print(f"\n{'='*80}")
        print(f"STEP {step}/{total_steps} (Epoch {(step-1)//steps_per_epoch + 1})")
        print(f"{'='*80}")
        
        # 1. RUN ADAM BATCH
        model_adam.train()
        try:
            data, target = next(adam_iter)
        except StopIteration:
            adam_iter = iter(adam_loader)
            data, target = next(adam_iter)
            
        data, target = data.to(device), target.to(device)
        optimizer_adam.zero_grad()
        output = model_adam(data)
        adam_loss = criterion(output, target)
        adam_loss.backward()
        optimizer_adam.step()
        
        adam_test_loss, adam_test_acc = fast_evaluate(model_adam, test_eval_loader, device, criterion)

        # 2. RUN ADAMCB BATCH
        model_adamcb.train()
        
        prev_w = adamcb_optimizer.w_t.copy()
        w_sum = np.sum(prev_w)
        prev_p = ((1.0 - GAMMA) * (prev_w / w_sum) + (GAMMA / DATASET_SIZE))
        
        adamcb_loss = adamcb_optimizer.step()
        
        sampled_indices = adamcb_optimizer.indices
        raw_grad_norms = adamcb_optimizer.gradient_norms.detach().cpu().numpy()
        new_w = adamcb_optimizer.w_t.copy()
        
        new_w_sum = np.sum(new_w)
        new_p = ((1.0 - GAMMA) * (new_w / new_w_sum) + (GAMMA / DATASET_SIZE))
        
        adamcb_test_loss, adamcb_test_acc = fast_evaluate(model_adamcb, test_eval_loader, device, criterion)
        
        # Track metrics
        adam_train_losses.append(adam_loss.item())
        adam_test_losses.append(adam_test_loss)
        adam_test_accs.append(adam_test_acc)
        
        adamcb_train_losses.append(adamcb_loss.item())
        adamcb_test_losses.append(adamcb_test_loss)
        adamcb_test_accs.append(adamcb_test_acc)
        
        # 3. PRINT COMPARISONS & TABLES
        print(f"[PERFORMANCE]")
        print(f"ADAM   -> Train Loss: {adam_loss.item():.4f} | Test Loss: {adam_test_loss:.4f} | Test Acc: {adam_test_acc:.2f}%")
        print(f"ADAMCB -> Train Loss: {adamcb_loss.item():.4f} | Test Loss: {adamcb_test_loss:.4f} | Test Acc: {adamcb_test_acc:.2f}%")
        print(f"-" * 80)
        
        print(f"[ADAMCB INTERNAL STATE - DATA POINTS SAMPLED IN BATCH]")
        df_dict = {}
        for i, idx in enumerate(sampled_indices):
            col_name = f"ID_{idx}"
            df_dict[col_name] = [
                float(prev_w[idx]),
                float(prev_p[idx]),
                float(raw_grad_norms[i]),
                float(new_w[idx]),
                float(new_w[idx] - prev_w[idx]),
                float(new_p[idx]),
                float(new_p[idx] - prev_p[idx])
            ]
            
        row_labels = [
            'Weight Before', 
            'Prob Before', 
            'Gradient Norm', 
            'Final Weight', 
            'Weight Change', 
            'Final Prob', 
            'Prob Change'
        ]
        
        df = pd.DataFrame(df_dict, index=row_labels).T
        df = df.sort_values(by=['Prob Before', 'Weight Before'], ascending=[False, False]).T
        df = df.map(lambda x: f"{float(x):.8f}")
        
        print(df)

    # ---------------------------------------------------------
    # 4. PLOT AND SAVE GRAPHS AT THE END OF TRAINING
    # ---------------------------------------------------------
    print("\nGenerating and saving comparison plots...")
    
    window_size = 20
    
    def smooth_data(data, window):
        # Calculate the average of every 'window' sized chunk
        return [np.mean(data[i:i+window]) for i in range(0, len(data), window)]

    # Calculate smoothed metrics
    adam_train_smooth = smooth_data(adam_train_losses, window_size)
    adamcb_train_smooth = smooth_data(adamcb_train_losses, window_size)
    
    adam_test_smooth = smooth_data(adam_test_losses, window_size)
    adamcb_test_smooth = smooth_data(adamcb_test_losses, window_size)
    
    adam_acc_smooth = smooth_data(adam_test_accs, window_size)
    adamcb_acc_smooth = smooth_data(adamcb_test_accs, window_size)
    
    # Generate X-axis representing the batch step at the end of each window
    x_axis = [min(i + window_size, len(adam_train_losses)) for i in range(0, len(adam_train_losses), window_size)]
    
    # --- Plot 1: Train Loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, adam_train_smooth, label='ADAM', alpha=0.8, linewidth=2, color='blue')
    plt.plot(x_axis, adamcb_train_smooth, label='ADAMCB', alpha=0.8, linewidth=2, color='orange')
    plt.xlabel('Step (Batch)')
    plt.ylabel(f'Train Loss (Avg over {window_size} steps)')
    plt.title(f'Train Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_loss_comparison.png')
    plt.close()

    # --- Plot 2: Test Loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, adam_test_smooth, label='ADAM', alpha=0.8, linewidth=2, color='blue')
    plt.plot(x_axis, adamcb_test_smooth, label='ADAMCB', alpha=0.8, linewidth=2, color='orange')
    plt.xlabel('Step (Batch)')
    plt.ylabel(f'Test Loss (Avg over {window_size} steps)')
    plt.title(f'Test Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('test_loss_comparison.png')
    plt.close()

    # --- Plot 3: Test Accuracy ---
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, adam_acc_smooth, label='ADAM', alpha=0.8, linewidth=2, color='blue')
    plt.plot(x_axis, adamcb_acc_smooth, label='ADAMCB', alpha=0.8, linewidth=2, color='orange')
    plt.xlabel('Step (Batch)')
    plt.ylabel(f'Test Accuracy (%) (Avg over {window_size} steps)')
    plt.title(f'Test Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('test_accuracy_comparison.png')
    plt.close()
    
    print("Training complete! Saved 'train_loss_comparison.png', 'test_loss_comparison.png', and 'test_accuracy_comparison.png'.")