import torch
import torch.nn as nn
import numpy as np
import json
import math

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Using device:", device)

# ==========================================
# HYPERPARAMETERS & CONFIGURATION
# ==========================================
N_neurons = 150
K_batch = 100
CHUNK_SIZE = 100  # Avoid OOM on MPS by processing in chunks
learning_rate = 0.02
num_epochs = 1000
module_size = 10
p_in = 0.35    # Dense local connections
p_out = 0.015  # Very sparse global connections
sparsity_penalty = 0.001
pruning_ratio = 0.03
prune_every = 100
num_top_k = 20
O_notes = 37
# ==========================================

# Notes mapping: 3 octaves, C3 to C6 (37 notes)
notes_map = {
    'C3': 0, 'C#3': 1, 'D3': 2, 'D#3': 3, 'E3': 4, 'F3': 5, 'F#3': 6, 'G3': 7, 'G#3': 8, 'A3': 9, 'A#3': 10, 'B3': 11,
    'C4': 12, 'C#4': 13, 'D4': 14, 'D#4': 15, 'E4': 16, 'F4': 17, 'F#4': 18, 'G4': 19, 'G#4': 20, 'A4': 21, 'A#4': 22, 'B4': 23,
    'C5': 24, 'C#5': 25, 'D5': 26, 'D#5': 27, 'E5': 28, 'F5': 29, 'F#5': 30, 'G5': 31, 'G#5': 32, 'A5': 33, 'A#5': 34, 'B5': 35,
    'C6': 36
}

# The sequence (1 step = 1 eighth note)
# Total 48 steps
seq = [
    ('G4', 1), ('G4', 1),
    ('A4', 2), ('G4', 2), ('C5', 2),
    ('B4', 4),
    ('G4', 1), ('G4', 1),
    ('A4', 2), ('G4', 2), ('D5', 2),
    ('C5', 4),
    ('G4', 1), ('G4', 1),
    ('G5', 2), ('E5', 2), ('C5', 2),
    ('B4', 2), ('A4', 2),
    ('F5', 1), ('F5', 1),
    ('E5', 2), ('C5', 2), ('D5', 2),
    ('C5', 4)  # total: 2+6+4+2+6+4+2+6+4+2+6+4 = Wait: 2+2+2+2+4+2+2+2+2+4 + 2+2+2+2+2 + 2+2+2+2+4 = 48
]

# Unroll sequence into list of notes per step
target_notes = []
for note, dur in seq:
    for i in range(dur):
        target_notes.append(notes_map[note])

T_steps = len(target_notes)

# Chords at each step (rough harmonization)
# Measures = 6 steps each (3/4 time, eighth steps)
# Meas 0 (pickup, step 0-1): C major (C4, E4, G4)
# Meas 1 (ste 2-7): C major (C4, E4, G4)
# Meas 2 (step 8-13): G major (G4, B4, D5)
# Meas 3 (step 14-19): G major (G4, B4, D5)
# Meas 4 (step 20-25): C major (C4, E4, G4)
# Meas 5 (step 26-31): C major (C4, E4, G4)
# Meas 6 (step 32-37): F major (F4, A4, C5)
# Meas 7 (step 38-43): C major (C4, E4, G4) -> C, G (let's do step 38-40 C, 41-43 G)
# Meas 8 (step 44-49): C major

chords = [
    ['C4','E4','G4'], ['C4','E4','G4'], # 0-1
    ['C4','E4','G4'], ['C4','E4','G4'], ['C4','E4','G4'], ['C4','E4','G4'], ['C4','E4','G4'], ['C4','E4','G4'], # 2-7
    ['G3','B3','D4'], ['G3','B3','D4'], ['G3','B3','D4'], ['G3','B3','D4'], ['G3','B3','D4'], ['G3','B3','D4'], # 8-13
    ['G3','B3','D4'], ['G3','B3','D4'], ['G3','B3','D4'], ['G3','B3','D4'], ['G3','B3','D4'], ['G3','B3','D4'], # 14-19
    ['C4','E4','G4'], ['C4','E4','G4'], ['C4','E4','G4'], ['C4','E4','G4'], ['C4','E4','G4'], ['C4','E4','G4'], # 20-25
    ['C4','E4','G4'], ['C4','E4','G4'], ['C4','E4','G4'], ['C4','E4','G4'], ['C4','E4','G4'], ['C4','E4','G4'], # 26-31
    ['F4','A4','C5'], ['F4','A4','C5'], ['F4','A4','C5'], ['F4','A4','C5'], ['F4','A4','C5'], ['F4','A4','C5'], # 32-37
    ['C4','E4','G4'], ['C4','E4','G4'], ['C4','E4','G4'], ['G3','B3','D4'], ['G3','B3','D4'], ['G3','B3','D4'], # 38-43
    ['C4','E4','G4'], ['C4','E4','G4'], ['C4','E4','G4'], ['C4','E4','G4'] # 44-47
]

T_target = torch.zeros((T_steps, O_notes))
for t in range(T_steps):
    # Base target: 1.0 for the melody note
    T_target[t, target_notes[t]] = 1.0
    
# Temporal smoothing (loss smaller if note is temporally close)
T_soft = T_target.clone()


for t in range(T_steps):
    for i in range(O_notes):
        # find distance to nearest t' where note i is played
        dists = [abs(t - t_prime) for t_prime in range(T_steps) if target_notes[t_prime] == i]
        if dists:
            min_dist = min(dists)
            if min_dist > 0:
                T_soft[t, i] = max(T_soft[t, i], math.exp(-min_dist / 2.0))


# Add chord bonus

for t in range(T_steps):
    for c_note in chords[t]:
        if c_note in notes_map:
            n_idx = notes_map[c_note]
            T_soft[t, n_idx] = max(T_soft[t, n_idx], 0.3)


T_soft = T_soft.to(device)

num_chunks = math.ceil(K_batch / CHUNK_SIZE)
print(f"Sampling {K_batch} networks across {num_chunks} chunks of {CHUNK_SIZE}...")

# Forward pass logic
def forward_pass(W, b, h):
    outputs = []
    current_h = h
    
    for t in range(T_steps):
        current_h_unsq = current_h.unsqueeze(1) # (K, 1, N)
        next_h = torch.bmm(current_h_unsq, W).squeeze(1) + b # (K, N)
        
        # Layer Normalization for all neurons
        mean_h = next_h.mean(dim=1, keepdim=True)
        var_h = next_h.var(dim=1, keepdim=True, unbiased=False)
        next_h = (next_h - mean_h) / torch.sqrt(var_h + 1e-5)
        
        current_h = torch.tanh(next_h)
        outputs.append(current_h)
        
    return torch.stack(outputs, dim=1) # (K, T, N)

def compute_loss(outputs):
    # Outputs: (K, T, N)
    # The first O_notes neurons are the output notes
    # Activation in [-1, 1]. Let's say we want it to match T_soft which is in [0, 1]
    # Map output to [0, 1] using (tanh + 1) / 2
    out_probs = (outputs[:, :, :O_notes] + 1) / 2.0
    
    # Non-output neurons should probably not explode
    # We want them to have some activity, but not saturate at 1 or -1 all the time
    
    # Main loss: MSE with T_soft
    # T_soft is (T, O_notes)
    # out_probs is (K, T, O_notes)
    target = T_soft.unsqueeze(0).expand(outputs.shape[0], T_steps, O_notes)
    mse = torch.mean((out_probs - target)**2, dim=(1,2))
    
    # Penalty for too many notes played at once
    # We sum the output activations at each step
    # Target sum is roughly 1 (melody) + maybe 1-2 chord notes ~ 2
    out_sums = torch.sum(out_probs, dim=2) # (K, T)
    sparsity_loss = torch.mean((out_sums - 1.5)**2, dim=1)
    
    return mse + sparsity_penalty * sparsity_loss

global_top_networks = []

for chunk_idx in range(num_chunks):
    actual_chunk = min(CHUNK_SIZE, K_batch - chunk_idx * CHUNK_SIZE)
    print(f"\n--- Processing Chunk {chunk_idx+1}/{num_chunks} ({actual_chunk} networks) ---")
    
    num_modules = N_neurons // module_size
    
    rand_tensor = torch.rand(actual_chunk, N_neurons, N_neurons, device=device)
    mask = (rand_tensor < p_out).float()
    
    for m in range(num_modules):
        start = m * module_size
        end = start + module_size
        block_rand = torch.rand(actual_chunk, module_size, module_size, device=device)
        block_mask = (block_rand < p_in).float()
        mask[:, start:end, start:end] = torch.max(mask[:, start:end, start:end], block_mask)

    W = torch.randn(actual_chunk, N_neurons, N_neurons, device=device) * mask * 0.5

    h0 = torch.randn(actual_chunk, N_neurons, device=device) * 0.1
    b = torch.randn(actual_chunk, N_neurons, device=device) * 0.05

    h0.requires_grad = True
    b.requires_grad = True
    W.requires_grad = True

    optimizer = torch.optim.Adam([h0, b, W], lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = forward_pass(W, b, h0)
        losses = compute_loss(outputs)
        
        loss = torch.mean(losses)
        best_loss = torch.min(losses)
        print(f"Epoch {epoch} - Avg Loss: {loss.item():.4f}, Best Loss: {best_loss.item():.4f}")
        
        loss.backward()
        
        W.grad *= mask
        optimizer.step()
        
        # Iterative Pruning Sequence
        if epoch > 0 and epoch % prune_every == 0:
            with torch.no_grad():
                mean_act = outputs.abs().mean(dim=1)
                mean_act[:, :O_notes] = float('inf')
                
                num_prune = int((N_neurons - O_notes) * pruning_ratio)
                _, prune_indices = torch.topk(mean_act, k=num_prune, largest=False, dim=1)
                
                for k in range(actual_chunk):
                    idx_to_prune = prune_indices[k]
                    W[k, idx_to_prune, :] = 0.0
                    W[k, :, idx_to_prune] = 0.0
                    b[k, idx_to_prune] = 0.0
                    h0[k, idx_to_prune] = 0.0
                    # Structurally obliterate paths locally out of mask entirely
                    mask[k, idx_to_prune, :] = 0.0
                    mask[k, :, idx_to_prune] = 0.0
    # Recompute final loss structurally accurately
    with torch.no_grad():
        final_outputs = forward_pass(W, b, h0)
        losses = compute_loss(final_outputs)

    k_local = min(num_top_k, actual_chunk)
    min_losses, min_indices = torch.topk(losses, k=k_local, largest=False)
    
    for loss_tensor, idx_tensor in zip(min_losses, min_indices):
        idx = idx_tensor.item()
        loss_val = loss_tensor.item()
        global_top_networks.append({
            'loss': loss_val,
            'W': W[idx].detach().cpu().numpy().tolist(),
            'b': b[idx].detach().cpu().numpy().tolist(),
            'h0': h0[idx].detach().cpu().numpy().tolist()
        })

# Sort all collected networks from all chunks and keep strictly top k
global_top_networks.sort(key=lambda x: x['loss'])
global_top_networks = global_top_networks[:num_top_k]

print(f"\n=== Global Top {num_top_k} Networks ===")
for rank, net in enumerate(global_top_networks):
    print(f"Rank {rank+1}: Loss = {net['loss']:.4f}")

data = {
    'networks': global_top_networks,
    'O_notes': O_notes,
    'T_steps': T_steps,
    'target_notes': target_notes,
    'notes_map': notes_map
}

with open('rnn_top20.json', 'w') as f:
    json.dump(data, f)
    
print("Saved to rnn_top20.json")
