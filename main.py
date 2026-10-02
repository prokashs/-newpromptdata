import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. Configuration
# ============================================================

batch_size = 32
block_size = 128       # maximum context length
max_iters = 3000
eval_interval = 300
learning_rate = 3e-4

n_embd = 128           # embedding dimension
n_head = 4             # attention heads
n_layer = 4            # transformer layers
dropout = 0.1

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 2. Training data
# ============================================================

# A small corpus.
# Replace this with a much larger text file for meaningful results.
text = """
Dog barks. 🐶 — Woof
Cow moos. 🐄 — Moo
Cat meows. 🐱 — Meow
Lion roars. 🦁 — Roar
Tiger growls. 🐯 — Growl
Horse neighs. 🐴 — Neigh
Sheep bleats. 🐑 — Baa
Goat bleats. 🐐 — Baa
Pig oinks. 🐷 — Oink
Duck quacks. 🦆 — Quack
Chicken clucks. 🐔 — Cluck
Rooster crows. 🐓 — Cock-a-doodle-doo
Donkey brays. 🫏 — Hee-haw
Elephant trumpets. 🐘 — Trumpet
Wolf howls. 🐺 — Howl
Snake hisses. 🐍 — Hiss
Frog croaks. 🐸 — Croak
Bee buzzes. 🐝 — Buzz
Bird chirps. 🐦 — Chirp
Owl hoots. 🦉 — Hoot
Crow caws. 🐦‍⬛ — Caw
Turkey gobbles. 🦃 — Gobble
Mouse squeaks. 🐭 — Squeak
Monkey chatters. 🐒 — Chatter
Gorilla grunts. 🦍 — Grunt
Bear growls. 🐻 — Growl
Deer bellows. 🦌 — Bellow
Dolphin clicks. 🐬 — Click
Whale sings. 🐋 — Song
Cricket chirps. 🦗 — Chirp
"""


# ============================================================
# 3. Character tokenizer
# ============================================================

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


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

    ix = torch.randint(
        len(source) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        source[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        source[i + 1:i + block_size + 1]
        for i in ix
    ])

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

        # Causal mask.
        # Token i cannot see future tokens.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # Attention scores
        wei = q @ k.transpose(-2, -1)

        # Scale attention
        wei = wei * (k.shape[-1] ** -0.5)

        # Causal masking
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # Convert scores to probabilities
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        # Weighted values
        v = self.value(x)

        out = wei @ v

        return out


# ============================================================
# 7. Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):

        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(
            num_heads * head_size,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )

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

        self.attention = MultiHeadAttention(
            n_head,
            head_size
        )

        self.feed_forward = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        # Attention + residual connection
        x = x + self.attention(
            self.ln1(x)
        )

        # Feed-forward + residual connection
        x = x + self.feed_forward(
            self.ln2(x)
        )

        return x


# ============================================================
# 10. GPT Model
# ============================================================

class MiniGPT(nn.Module):

    def __init__(self):

        super().__init__()

        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        # Positional embedding
        self.position_embedding = nn.Embedding(
            block_size,
            n_embd
        )

        # Transformer
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head)
                for _ in range(n_layer)
            ]
        )

        self.layer_norm = nn.LayerNorm(n_embd)

        # Convert embeddings into vocabulary logits
        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        # Token embeddings
        token_emb = self.token_embedding(idx)

        # Position embeddings
        positions = torch.arange(
            T,
            device=device
        )

        pos_emb = self.position_embedding(positions)

        # Combine them
        x = token_emb + pos_emb

        # Transformer
        x = self.blocks(x)

        # Final normalization
        x = self.layer_norm(x)

        # Vocabulary logits
        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            # Only use the most recent context
            idx_cond = idx[:, -block_size:]

            # Get predictions
            logits, _ = self(idx_cond)

            # Last token prediction
            logits = logits[:, -1, :]

            # Convert to probabilities
            probabilities = F.softmax(
                logits,
                dim=-1
            )

            # Sample next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Append token
            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

        return idx


# ============================================================
# 11. Create model
# ============================================================

model = MiniGPT().to(device)

print()
print("Number of parameters:",
      sum(p.numel() for p in model.parameters()))


# ============================================================
# 12. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# 13. Training
# ============================================================

for iteration in range(max_iters):

    # Get training batch
    xb, yb = get_batch("train")

    # Forward pass
    logits, loss = model(xb, yb)

    # Clear gradients
    optimizer.zero_grad(set_to_none=True)

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    if iteration % eval_interval == 0:

        print(
            f"iteration {iteration}: "
            f"loss {loss.item():.4f}"
        )


# ============================================================
# 14. Generate text
# ============================================================

print("\nGenerating...\n")

context = torch.zeros(
    (1, 1),
    dtype=torch.long,
    device=device
)

generated = model.generate(
    context,
    max_new_tokens=300
)

print(decode(generated[0].tolist()))
