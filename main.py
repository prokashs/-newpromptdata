import torch
import torch.nn as nn
import torch.nn.functional as F
import re

# ============================================================
# 1. Configuration
# ============================================================
batch_size = 32
block_size = 128  # maximum context length
max_iters = 3000
eval_interval = 300
learning_rate = 3e-4

n_embd = 128  # embedding dimension
n_head = 4  # attention heads
n_layer = 4  # transformer layers
dropout = 0.1

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Using device:", device)

# ============================================================
# 2. Training data (UPDATED: System Status & Codes Dataset)
# ============================================================
text = """
System evaluation status checklist. Safe operational code sequence is: 0 1 2 3 .
System evaluation status checklist. Safe operational code sequence is: 1 1 2 0 .
System evaluation status checklist. Safe operational code sequence is: 2 3 0 1 .
System evaluation status checklist. Safe operational code sequence is: 0 0 1 2 .
System evaluation status checklist. Safe operational code sequence is: 3 2 1 0 .
System evaluation status checklist. Safe operational code sequence is: 1 2 3 0 .
System evaluation status checklist. Safe operational code sequence is: 2 1 0 3 .
System evaluation status checklist. Safe operational code sequence is: 0 3 2 1 .
System evaluation status checklist. Safe operational code sequence is: 1 0 3 2 .
System evaluation status checklist. Safe operational code sequence is: 3 3 1 2 .
System evaluation status checklist. Safe operational code sequence is: 2 2 0 1 .
System evaluation status checklist. Safe operational code sequence is: 0 1 1 3 .
System evaluation status checklist. Safe operational code sequence is: 1 2 0 3 .
System evaluation status checklist. Safe operational code sequence is: 3 0 2 1 .
System evaluation status checklist. Safe operational code sequence is: 2 0 1 3 .
"""

# ============================================================
# 3. Character tokenizer
# ============================================================
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s if c in stoi]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
print("Vocabulary size:", vocab_size)
print("Dataset size:", len(data))

# ============================================================
# 4. Train / validation split
# ============================================================
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# ============================================================
# 5. Batch generation
# ============================================================
def get_batch(split):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ============================================================
# 6. Self Attention Head
# ============================================================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


# ============================================================
# 7. Multi-Head Attention
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        return self.dropout(out)


# ============================================================
# 8. Feed Forward Network
# ============================================================
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# 9. Transformer Block
# ============================================================
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


# ============================================================
# 10. GPT Model (with Guardrail Methods)
# ============================================================
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding(idx)
        positions = torch.arange(T, device=device)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    @torch.no_grad()
    def generate_safe_assertion(self, idx, allowed_chars, max_new_tokens=10):
        self.eval()
        allowed_indices = [stoi[ch] for ch in allowed_chars if ch in stoi]
        if not allowed_indices:
            raise ValueError("None of the allowed_chars exist in the model's vocabulary!")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :]

            mask = torch.full_like(next_token_logits, float('-inf'))
            mask[:, allowed_indices] = next_token_logits[:, allowed_indices]

            probabilities = F.softmax(mask, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            if next_token.item() == stoi.get('.', -1):
                break
        return idx


# ============================================================
# 11. Independent Content Filter Layer
# ============================================================
class ExecutionGuardrail:
    def __init__(self):
        self.blocked_patterns = [
            r"rm\s+-rf", r"<script>", r"eval\(", r"Authorization:"
        ]

    def verify_payload(self, generated_text):
        normalized_text = generated_text.strip()
        for pattern in self.blocked_patterns:
            if re.search(pattern, normalized_text, re.IGNORECASE):
                print(f"🚨 SECURITY ALERT: Guardrail tripped by pattern: '{pattern}'")
                return False
        return True


# ============================================================
# 12. Application Guardrail Execution Loop
# ============================================================
if __name__ == "__main__":
    guard = ExecutionGuardrail()

    print("\nInitializing MiniGPT infrastructure...")
    model = MiniGPT().to(device)

    print("\nTraining MiniGPT model on corpus patterns...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for iteration in range(max_iters):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iteration % 500 == 0:
            print(f"Iteration {iteration:4d} | Current Loss: {loss.item():.4f}")

    print("Training complete! Model weights optimized.\n")

    prompt = "System evaluation status checklist. Safe operational code sequence is:"
    context_tokens = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    print(f"Processing Prompt: '{prompt}'")
    print("Running through Neurosymbolic Guardrails...")

    safe_allowed_chars = ['0', '1', '2', '3', ' ', '.']
    output_tokens = model.generate_safe_assertion(
        context_tokens,
        allowed_chars=safe_allowed_chars,
        max_new_tokens=10
    )

   ### decoded_output = decode(output_tokens.tolist())
    decoded_output = decode(output_tokens[0].tolist())

    new_generation = decoded_output[len(prompt):]
    print(f"Layer 1 Token Mask Result: '{new_generation}'")

    if guard.verify_payload(new_generation):
        print("✅ Layer 2 Passed: Content meets safety specifications.")
        print(f"Executing system instructions with verified variable: {new_generation.strip()}")
    else:
        print("❌ CRITICAL VETO: Content blocked by firewall. Action dropped.")
