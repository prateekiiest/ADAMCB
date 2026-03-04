import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import pandas as pd
import random
from torch.func import functional_call, vmap, grad

# Display settings to ensure all columns (Idx_0 to Idx_49) print without truncating
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
        
        self.w_t = np.ones(self.n, dtype=np.float64)
        self.p_t = np.ones(self.n, dtype=np.float64) * (self.K / self.n)
        
    def verbose_depround(self, p_input):
        print("\n--- 7. DEPROUND ALGORITHM ITERATIONS ---")
        p = p_input.copy()
        epsilon = 1e-9
        iteration = 0
        
        # Store history to print as transposed DataFrame
        history = {f"Iter_{iteration} (Start)": p.copy()}
        
        while True:
            floating_indices = np.where((p > epsilon) & (p < 1.0 - epsilon))[0]
            if len(floating_indices) < 2:
                break
            
            i, j = np.random.choice(floating_indices, 2, replace=False)
            alpha = min(1.0 - p[i], p[j])
            beta = min(p[i], 1.0 - p[j])
            
            prob_scenario_1 = beta / (alpha + beta)
            if np.random.rand() < prob_scenario_1:
                p[i] += alpha; p[j] -= alpha
            else:
                p[i] -= beta; p[j] += beta
                
            p = np.clip(p, 0.0, 1.0)
            iteration += 1
            history[f"Iter_{iteration} (Upd {i},{j})"] = p.copy()
            
        indices = np.argsort(p)[-self.K:]
        
        # Print transposed history
        df_depround = pd.DataFrame(history).T
        df_depround.columns = [f"Idx_{idx}" for idx in range(self.n)]
        print(df_depround.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"\nFinal Selected Indices: {indices}")
        
        return indices

    def step(self, full_dataset, loss_fn):
        self.t += 1
        w_prev = self.w_t.copy()
        p_prev = self.p_t.copy()
        
        # --- ALGORITHM 2: BATCH SELECTION ---
        C = (1.0 / self.K - self.gamma / self.n) / (1.0 - self.gamma)
        
        print("\n--- 5. ALGORITHM 2 (BATCH SELECTION) VARIABLES & 9. CALCULATIONS ---")
        print(f"C calculation = (1/K - gamma/n) / (1 - gamma) = (1/{self.K} - {self.gamma}/{self.n}) / (1 - {self.gamma}) = {C:.6f}")
        
        theSum = np.sum(w_prev)
        max_w = np.max(w_prev)
        threshold_check = C * theSum
        print(f"Max Weight Check: max(w) = {max_w:.4f}, C * sum(w) = {threshold_check:.4f}")
        
        w_temp = w_prev.copy()
        tau = 1.0
        S_null = []
        
        if max_w >= threshold_check:
            w_sorted = np.sort(w_temp)[::-1]
            S_val = np.sum(w_sorted)
            for i in range(1, len(w_sorted) + 1):
                tau = (C * S_val) / (1.0 - i * C)
                if tau > w_sorted[i-1]:
                    break
                S_val -= w_sorted[i-1]
                
            S_null = np.where(w_temp >= tau)[0].tolist()
            w_temp[S_null] = tau
            
        print(f"Final Tau (T) = {tau:.6f} | S_null = {S_null}")
        
        # --- 6. WEIGHT CLIPPING (TRANSPOSED) ---
        print("\n--- 6. WEIGHT CLIPPING W(i,t) vs W'(i,t) ---")
        df_clip = pd.DataFrame({"Original W(i,t)": w_prev, "Clipped W'(i,t)": w_temp}).T
        df_clip.columns = [f"Idx_{idx}" for idx in range(self.n)]
        print(df_clip.to_string(float_format=lambda x: f"{x:.4f}"))
        
        w_sum = np.sum(w_temp)
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
        raw_grad_norms = torch.sqrt(per_sample_norm_sq).detach().cpu().numpy()
        
        scaling_factor = 1.0 / (self.n * self.p_t[J_t])
        G_t_hat = []
        for name, param in self.model.named_parameters():
            scaled_grad = sample_grads[name] * torch.tensor(scaling_factor, device=self.device, dtype=torch.float32).view(-1, *([1]*(sample_grads[name].dim()-1)))
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
            
        # --- ALGORITHM 3: WEIGHT UPDATE ---
        print("\n--- 8. WEIGHT UPDATION VARIABLES & 9. CALCULATIONS ---")
        p_min = self.gamma / self.n
        L = np.max(raw_grad_norms) + 1e-8
        print(f"Constants: K={self.K}, gamma={self.gamma}, p_min={p_min:.6f}, L={L:.6f}")
        
        g_norms_all = np.full(self.n, np.nan)
        l_jt_all = np.full(self.n, np.nan)
        
        for batch_idx, j in enumerate(J_t):
            norm_g = raw_grad_norms[batch_idx]
            prob_j = self.p_t[j]
            
            term1 = - (norm_g**2) / (prob_j**2)
            term2 = (L**2) / (p_min**2)
            l_j_t = (p_min**2 / L**2) * (term1 + term2)
            
            g_norms_all[j] = norm_g
            l_jt_all[j] = l_j_t
            
            print(f"  Calc Idx {j}: l_j_t = ({p_min:.4f}^2 / {L:.4f}^2) * (-{norm_g:.4f}^2 / {prob_j:.4f}^2 + {L:.4f}^2 / {p_min:.4f}^2) = {l_j_t:.6f}")
            
            if j not in S_null:
                exponent = -self.K * self.gamma * l_j_t / self.n
                self.w_t[j] = self.w_t[j] * np.exp(exponent)
                
        df_weight_upd = pd.DataFrame({"g(i,t) [||g||]": g_norms_all, "l(j,t)": l_jt_all}).T
        df_weight_upd.columns = [f"Idx_{idx}" for idx in range(self.n)]
        print(df_weight_upd.fillna('-').to_string(float_format=lambda x: f"{x:.6f}"))

        # --- PRINTING POINTS 1-4: DATAPOINT TRACKING (TRANSPOSED) ---
        print("\n--- 1, 2, 3, 4. PER-DATAPOINT TRACKING (ALL N POINTS) ---")
        df_dict = {
            "1. Prev_Weight": w_prev, 
            "2. New_Weight": self.w_t, 
            "2. Weight_Change": self.w_t - w_prev,
            "3. Prev_Prob": p_prev, 
            "4. New_Prob": self.p_t, 
            "4. Prob_Change": self.p_t - p_prev
        }
        df = pd.DataFrame(df_dict).T
        df.columns = [f"Idx_{i}" for i in range(self.n)]
        print(df.to_string(float_format=lambda x: f"{x:.6f}"))
        
        return loss_val, raw_grad_norms.min(), raw_grad_norms.max(), raw_grad_norms.mean(), J_t
        
# -------------------------------------------------------------------------
# EXECUTION SCRIPT
# -------------------------------------------------------------------------
if __name__ == '__main__':
    set_seed(42)
    device = torch.device("cpu") 
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_subset = IndexedSubset(Subset(full_dataset, range(50)))
    test_subset = Subset(test_dataset, range(50))
    
    BATCH_SIZE = 8
    EPOCHS = 10  
    total_steps = len(train_subset) // BATCH_SIZE
    
    adam_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    model_adam = MLP().to(device)
    model_cb = MLP().to(device)
    model_cb.load_state_dict(model_adam.state_dict())
    
    opt_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001)
    opt_cb = AdamCBOptimizerVerbose(model_cb, n=50, K=BATCH_SIZE, lr=0.001, gamma=0.1, device=device)
    
    def evaluate(model):
        model.eval()
        loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                loss += criterion(model(data), target).item() * data.size(0)
        return loss / 50
        
    adam_iter = iter(adam_loader)
    for i in range(EPOCHS):
        print(f"Training Epoch {i+1} =>>\n")
        for step in range(1, total_steps + 1):
            print(f"\n{'='*30} STARTING BATCH {step}/{total_steps} {'='*30}")
        
            # --- ADAM RUN ---
            model_adam.train()
        
            # Handle the iterator correctly across multiple epochs
            try:
                idx_adam, data, target = next(adam_iter)
            except StopIteration:
                adam_iter = iter(adam_loader)
                idx_adam, data, target = next(adam_iter)
        
            params = dict(model_adam.named_parameters())
            buffers = dict(model_adam.named_buffers())
            def compute_loss_adam(p, b, s, t): 
                return criterion(functional_call(model_adam, (p, b), (s.unsqueeze(0),)), t.unsqueeze(0))
        
            adam_grads = vmap(grad(compute_loss_adam), in_dims=(None, None, 0, 0))(params, buffers, data, target)
        
            # FIX: Use data.size(0) to handle the smaller final batch
            current_batch_size = data.size(0)
            adam_norm_sq = torch.zeros(current_batch_size, device=device)
        
            for name, param in model_adam.named_parameters():
                adam_norm_sq += (adam_grads[name].flatten(start_dim=1) ** 2).sum(dim=1)
            adam_norms = torch.sqrt(adam_norm_sq).detach().cpu().numpy()
        
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
            print(f"ADAM   | Unique Datapoints: {idx_adam.tolist()}")
            print(f"ADAM   | Batch Grads -> Min: {adam_norms.min():.4f}, Max: {adam_norms.max():.4f}, Mean: {adam_norms.mean():.4f}")
            print("-" * 50)
            print(f"ADAMCB | Train Loss: {loss_cb:.4f} | Test Loss: {test_loss_cb:.4f}")
            print(f"ADAMCB | Unique Datapoints: {J_t.tolist()}")
            print(f"ADAMCB | Batch Grads -> Min: {min_g:.4f}, Max: {max_g:.4f}, Mean: {mean_g:.4f}")