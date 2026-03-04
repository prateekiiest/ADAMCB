import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import pandas as pd
import random
from torch.func import functional_call, vmap, grad
import matplotlib.pyplot as plt

# Keep global max_columns limited so the 30,000 full-dataset prints don't freeze your terminal.
# We will use 'with pd.option_context' to print the 128 batch items fully.
pd.set_option('display.max_columns', 10) 
pd.set_option('display.width', 2000)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

class AdamFromScratch:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        
        # Initialize moment tensors on the same device as the parameters
        self.m = [torch.zeros_like(p.data) for p in self.parameters]
        self.v = [torch.zeros_like(p.data) for p in self.parameters]

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self):
        """Performs a single optimization step."""
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
                
            grad = p.grad.data
            
            # Update biased first moment estimate
            self.m[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            # Update biased second raw moment estimate
            self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            
            # Compute bias-corrected first and second moment estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p.data.addcdiv_(m_hat, torch.sqrt(v_hat).add_(self.eps), value=-self.lr)


class AdamCBOptimizerVerbose:
    def __init__(self, model, n, K, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, gamma=0.1, device='cpu'):
        self.model = model
        self.n = n
        self.K = K
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.gamma = gamma
        self.device = device

        self.m = {name: torch.zeros_like(p.data) for name, p in self.model.named_parameters()}
        self.v = {name: torch.zeros_like(p.data) for name, p in self.model.named_parameters()}
        self.t = 0
        
        self.w_t = torch.ones(self.n, dtype=torch.float64, device=self.device)
        self.p_t = torch.ones(self.n, dtype=torch.float64, device=self.device) * (self.K / self.n)
        
    def verbose_depround(self, p_input):
        print("\n--- 7. DEPROUND ALGORITHM ---")
        p = p_input.clone()
        epsilon = 1e-9
        
        # Only store Step_0 and Step_K. Using detach() to ensure no gradient graphs are attached.
        history = {"Step_0 (Before)": p.detach().cpu().numpy().copy()}
        
        while True:
            floating_mask = (p > epsilon) & (p < 1.0 - epsilon)
            floating_indices = torch.nonzero(floating_mask).squeeze(1)
            
            if len(floating_indices) < 2:
                break
            
            num_pairs = min(64, len(floating_indices) // 2)
            
            perm = torch.randperm(len(floating_indices), device=self.device)
            selected_idx = floating_indices[perm[:2 * num_pairs]]
            
            i_idx = selected_idx[:num_pairs]
            j_idx = selected_idx[num_pairs:]
            
            p_i = p[i_idx]
            p_j = p[j_idx]
            
            alpha = torch.minimum(1.0 - p_i, p_j)
            beta = torch.minimum(p_i, 1.0 - p_j)
            
            prob_scenario_1 = beta / (alpha + beta)
            rand_vals = torch.rand(num_pairs, dtype=torch.float64, device=self.device)
            
            scenario_1_mask = rand_vals < prob_scenario_1
            
            p_i_new = torch.where(scenario_1_mask, p_i + alpha, p_i - beta)
            p_j_new = torch.where(scenario_1_mask, p_j - alpha, p_j + beta)
            
            p[i_idx] = p_i_new
            p[j_idx] = p_j_new
            p = torch.clamp(p, 0.0, 1.0)
            
        history["Step_K (After)"] = p.detach().cpu().numpy().copy()
            
        _, indices_tensor = torch.topk(p, self.K)
        indices = indices_tensor.detach().cpu().numpy()
        
        df_depround = pd.DataFrame(history).T
        df_depround.columns = [f"Idx_{idx}" for idx in range(self.n)]
        print(df_depround.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"\nFinal Selected Indices: {indices[:10]}... (truncated)")
        
        return indices

    def step(self, full_dataset, loss_fn):
        self.t += 1
        w_prev = self.w_t.clone()
        p_prev = self.p_t.clone()
        
        # --- ALGORITHM 2: BATCH SELECTION ---
        C = (1.0 / self.K - self.gamma / self.n) / (1.0 - self.gamma)
        
        print("\n--- 5. ALGORITHM 2 (BATCH SELECTION) VARIABLES & 9. CALCULATIONS ---")
        print(f"C calculation = (1/K - gamma/n) / (1 - gamma) = (1/{self.K} - {self.gamma}/{self.n}) / (1 - {self.gamma}) = {C:.6f}")
        
        theSum = torch.sum(w_prev).item()
        max_w = torch.max(w_prev).item()
        threshold_check = C * theSum
        print(f"Max Weight Check: max(w) = {max_w:.4f}, C * sum(w) = {threshold_check:.4f}")
        
        w_temp = w_prev.clone()
        tau = 1.0
        S_null = []
        
        if max_w >= threshold_check:
            w_sorted = torch.sort(w_temp, descending=True)[0]
            S_val = torch.sum(w_sorted).item()
            for i in range(1, len(w_sorted) + 1):
                tau = (C * S_val) / (1.0 - i * C)
                if tau > w_sorted[i-1].item():
                    break
                S_val -= w_sorted[i-1].item()
                
            S_null_mask = w_temp >= tau
            S_null = torch.nonzero(S_null_mask).squeeze(1).detach().cpu().tolist()
            w_temp[S_null_mask] = tau
            
        print(f"Final Tau (T) = {tau:.6f} | S_null = {S_null[:10]}... (truncated for display)")
        
        # --- 6. WEIGHT CLIPPING (TRANSPOSED) ---
        print("\n--- 6. WEIGHT CLIPPING W(i,t) vs W'(i,t) ---")
        df_clip = pd.DataFrame({"Original W(i,t)": w_prev.detach().cpu().numpy(), "Clipped W'(i,t)": w_temp.detach().cpu().numpy()}).T
        df_clip.columns = [f"Idx_{idx}" for idx in range(self.n)]
        print(df_clip.to_string(float_format=lambda x: f"{x:.4f}"))
        
        w_sum = torch.sum(w_temp)
        self.p_t = self.K * ((1.0 - self.gamma) * (w_temp / w_sum) + (self.gamma / self.n))
        
        # --- DEPROUND ---
        J_t = self.verbose_depround(self.p_t)
        
        # --- COMPUTE GRADIENTS ---
        batch_items = [full_dataset[k] for k in J_t]
        data_batch = torch.stack([item[1] for item in batch_items]).to(self.device)
        target_batch = torch.tensor([item[2] for item in batch_items], device=self.device)
        
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        
        def compute_loss(params, buffers, sample, target):
            pred = functional_call(self.model, (params, buffers), (sample.unsqueeze(0),))
            return loss_fn(pred, target.unsqueeze(0))
            
        sample_grads = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))(params, buffers, data_batch, target_batch)
        
        per_sample_norm_sq = torch.zeros(self.K, device=self.device)
        for name, param in self.model.named_parameters():
            per_sample_norm_sq += (sample_grads[name].flatten(start_dim=1) ** 2).sum(dim=1)
        raw_grad_norms = torch.sqrt(per_sample_norm_sq) 
        
        scaling_factor = 1.0 / (self.n * self.p_t[J_t])
        G_t_hat = []
        for name, param in self.model.named_parameters():
            scaled_grad = sample_grads[name] * scaling_factor.view(-1, *([1]*(sample_grads[name].dim()-1))).to(torch.float32)
            G_t_hat.append(scaled_grad.sum(dim=0) / self.K)
            
        bias1, bias2 = 1 - self.beta1**self.t, 1 - self.beta2**self.t
        for i, (name, param) in enumerate(self.model.named_parameters()):
            g = G_t_hat[i]
            self.m[name].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            self.v[name].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            m_hat, v_hat = self.m[name] / bias1, self.v[name] / bias2
            param.data.addcdiv_(m_hat, torch.sqrt(v_hat).add_(self.eps), value=-self.lr)

        with torch.no_grad():
            output = self.model(data_batch)
            loss_val = loss_fn(output, target_batch).item()
            
        # --- ALGORITHM 3: WEIGHT UPDATE (PARALLEL ON GPU) ---
        print("\n--- 8. WEIGHT UPDATION VARIABLES & 9. CALCULATIONS ---")
        p_min = self.gamma / self.n
        L = torch.max(raw_grad_norms).item() + 1e-8
        print(f"Constants: K={self.K}, gamma={self.gamma}, p_min={p_min:.6f}, L={L:.6f}")
        
        prob_j_tensor = self.p_t[J_t]
        term1 = - (raw_grad_norms**2) / (prob_j_tensor**2)
        term2 = (L**2) / (p_min**2)
        l_j_t_tensor = (p_min**2 / L**2) * (term1 + term2)
        
        # .detach() explicitly ensures there are no graph remnants when converting
        g_norms_cpu = raw_grad_norms.detach().cpu().numpy()
        l_jt_cpu = l_j_t_tensor.detach().cpu().numpy()
        prob_j_cpu = prob_j_tensor.detach().cpu().numpy()
        
        # Sort values by index before creating the DataFrame
        sort_order = np.argsort(J_t)
        sorted_J_t = J_t[sort_order]
        
        # Transpose to print horizontally
        batch_df = pd.DataFrame({
            "g(j,t)": g_norms_cpu[sort_order],
            "p(j,t)": prob_j_cpu[sort_order],
            "l(j,t)": l_jt_cpu[sort_order]
        }).T
        batch_df.columns = [f"Idx_{idx}" for idx in sorted_J_t]
        
        print("\n--- ADAMCB BATCH DATAPOINTS: g(j,t), p(j,t), l(j,t) (SORTED, HORIZONTAL) ---")
        # Temporarily allow infinite columns so the batch isn't broken into chunks vertically
        with pd.option_context('display.max_columns', None):
            print(batch_df.to_string(float_format=lambda x: f"{x:.6f}"))

        S_null_set = set(S_null)
        valid_mask = torch.tensor([j not in S_null_set for j in J_t], device=self.device)
        valid_indices = torch.tensor(J_t, device=self.device)[valid_mask]
        
        exponent = -self.K * self.gamma * l_j_t_tensor[valid_mask] / self.n
        self.w_t[valid_indices] *= torch.exp(exponent)
                
        # --- PRINTING POINTS 1-4: DATAPOINT TRACKING (TRANSPOSED) ---
        print("\n--- 1, 2, 3, 4. PER-DATAPOINT TRACKING (ALL N POINTS) ---")
        w_prev_cpu = w_prev.detach().cpu().numpy()
        w_t_cpu = self.w_t.detach().cpu().numpy()
        p_prev_cpu = p_prev.detach().cpu().numpy()
        p_t_cpu = self.p_t.detach().cpu().numpy()
        
        df_dict = {
            "1. Prev_Weight": w_prev_cpu, 
            "2. New_Weight": w_t_cpu, 
            "2. Weight_Change": w_t_cpu - w_prev_cpu,
            "3. Prev_Prob": p_prev_cpu, 
            "4. New_Prob": p_t_cpu, 
            "4. Prob_Change": p_t_cpu - p_prev_cpu
        }
        df = pd.DataFrame(df_dict).T
        df.columns = [f"Idx_{i}" for i in range(self.n)]
        print(df.to_string(float_format=lambda x: f"{x:.6f}"))
        
        return loss_val, g_norms_cpu.min(), g_norms_cpu.max(), g_norms_cpu.mean(), J_t

# -------------------------------------------------------------------------
# EXECUTION SCRIPT
# -------------------------------------------------------------------------
if __name__ == '__main__':
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # 30K and 1K
    train_subset = IndexedSubset(Subset(full_dataset, range(30000)))
    test_subset = Subset(test_dataset, range(1000))
    
    BATCH_SIZE = 128
    EPOCHS = 5  
    total_steps = len(train_subset) // BATCH_SIZE
    
    adam_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    model_adam = MLP().to(device)
    model_cb = MLP().to(device)
    model_cb.load_state_dict(model_adam.state_dict())
    
    opt_adam = AdamFromScratch(model_adam.parameters(), lr=0.001)
    # Using 30000 for n and 128 for K
    opt_cb = AdamCBOptimizerVerbose(model_cb, n=30000, K=BATCH_SIZE, lr=0.001, gamma=0.1, device=device)
    
    def evaluate(model):
        model.eval()
        loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                loss += criterion(model(data), target).item() * data.size(0)
        return loss / 1000
        
    adam_iter = iter(adam_loader)
    
    # Tracking arrays for plotting
    plot_steps = []
    adam_train_losses, adam_test_losses = [], []
    cb_train_losses, cb_test_losses = [], []
    
    step_counter = 0

    for i in range(EPOCHS):
        print(f"\n\n{'*'*50}\nTraining Epoch {i+1} =>>\n{'*'*50}")
        for step in range(1, total_steps + 1):
            step_counter += 1
            print(f"\n{'='*30} STARTING BATCH {step}/{total_steps} {'='*30}")
        
            # --- ADAM RUN ---
            model_adam.train()
        
            try:
                idx_adam, data, target = next(adam_iter)
            except StopIteration:
                adam_iter = iter(adam_loader)
                idx_adam, data, target = next(adam_iter)
            
            data, target = data.to(device), target.to(device)
        
            params = dict(model_adam.named_parameters())
            buffers = dict(model_adam.named_buffers())
            def compute_loss_adam(p, b, s, t): 
                return criterion(functional_call(model_adam, (p, b), (s.unsqueeze(0),)), t.unsqueeze(0))
        
            adam_grads = vmap(grad(compute_loss_adam), in_dims=(None, None, 0, 0))(params, buffers, data, target)
        
            current_batch_size = data.size(0)
            adam_norm_sq = torch.zeros(current_batch_size, device=device)
        
            for name, param in model_adam.named_parameters():
                adam_norm_sq += (adam_grads[name].flatten(start_dim=1) ** 2).sum(dim=1)
            
            # Use detach() to prevent graph history errors
            adam_norms = torch.sqrt(adam_norm_sq).detach().cpu().numpy()
            idx_adam_cpu = idx_adam.detach().cpu().numpy()
        
            # Calculate l(j,t) for Adam
            adam_p_j = BATCH_SIZE / 30000.0
            adam_p_min = 0.1 / 30000.0  # gamma / n
            adam_L = adam_norms.max() + 1e-8
            
            adam_term1 = - (adam_norms**2) / (adam_p_j**2)
            adam_term2 = (adam_L**2) / (adam_p_min**2)
            adam_l_jt = (adam_p_min**2 / adam_L**2) * (adam_term1 + adam_term2)
            
            # Sort values by index before creating the DataFrame
            sort_order = np.argsort(idx_adam_cpu)
            sorted_idx = idx_adam_cpu[sort_order]
            adam_p_j_array = np.array([adam_p_j] * len(adam_norms))
            
            # Transpose to print horizontally
            adam_batch_df = pd.DataFrame({
                "g(j,t)": adam_norms[sort_order],
                "p(j,t)": adam_p_j_array[sort_order],
                "l(j,t)": adam_l_jt[sort_order]
            }).T
            adam_batch_df.columns = [f"Idx_{idx}" for idx in sorted_idx]
            
            print("\n--- ADAM BATCH DATAPOINTS: g(j,t), p(j,t), l(j,t) (SORTED, HORIZONTAL) ---")
            with pd.option_context('display.max_columns', None):
                print(adam_batch_df.to_string(float_format=lambda x: f"{x:.6f}"))

            opt_adam.zero_grad()
            loss_adam = criterion(model_adam(data), target)
            loss_adam.backward()
            opt_adam.step()
            test_loss_adam = evaluate(model_adam)
        
            # --- ADAMCB RUN ---
            model_cb.train()
            loss_cb, min_g, max_g, mean_g, J_t = opt_cb.step(train_subset, criterion)
            test_loss_cb = evaluate(model_cb)
        
            print("\n--- SUMMARY OF METRICS FOR BATCH ---")
            print(f"ADAM   | Train Loss: {loss_adam.item():.4f} | Test Loss: {test_loss_adam:.4f}")
            print(f"ADAM   | Unique Datapoints: {idx_adam_cpu.tolist()[:10]}... (truncated)")
            print(f"ADAM   | Batch Grads -> Min: {adam_norms.min():.4f}, Max: {adam_norms.max():.4f}, Mean: {adam_norms.mean():.4f}")
            print("-" * 50)
            print(f"ADAMCB | Train Loss: {loss_cb:.4f} | Test Loss: {test_loss_cb:.4f}")
            print(f"ADAMCB | Unique Datapoints: {J_t.tolist()[:10]}... (truncated)")
            print(f"ADAMCB | Batch Grads -> Min: {min_g:.4f}, Max: {max_g:.4f}, Mean: {mean_g:.4f}")
            
            # Store metrics
            plot_steps.append(step_counter)
            adam_train_losses.append(loss_adam.item())
            adam_test_losses.append(test_loss_adam)
            cb_train_losses.append(loss_cb)
            cb_test_losses.append(test_loss_cb)

    # --- PLOTTING ---
    plt.figure(figsize=(14, 6))

    # Train Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(plot_steps, adam_train_losses, label='Adam Train Loss', alpha=0.7)
    plt.plot(plot_steps, cb_train_losses, label='AdamCB Train Loss', alpha=0.7)
    plt.title('Train Loss: Adam vs AdamCB')
    plt.xlabel('Batch Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Test Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(plot_steps, adam_test_losses, label='Adam Test Loss', alpha=0.7)
    plt.plot(plot_steps, cb_test_losses, label='AdamCB Test Loss', alpha=0.7)
    plt.title('Test Loss: Adam vs AdamCB')
    plt.xlabel('Batch Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('adam_vs_adamcb_loss_graphs.png')
    plt.show()
    print("\nTraining complete. Loss graphs have been saved as 'adam_vs_adamcb_loss_graphs.png' and displayed.")