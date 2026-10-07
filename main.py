
import torch
print("PyTorch version:", torch.__version__)
print("Tensor test:", torch.tensor([1, 2, 3]))

import torch.nn as nn
import torch.nn.functional as F
import re

# ============================================================
# 1. Configuration
# ============================================================
batch_size = 32
block_size = 128
max_iters = 3000
learning_rate = 3e-4

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 2. Training Data
# ============================================================
# The model is learning character-by-character.
#
# The important relationships are:
#
#     Dog  -> barks
#     Cat  -> meows
#
# We repeat the examples so that this tiny training corpus
# provides enough samples for the model to learn the pattern.
# ============================================================

text = """
Dog barks.
Cat meows.
Dog barks.
Cat meows.
Dog barks loudly.
Cat meows softly.
A dog barks.
A cat meows.
The dog barks.
The cat meows.
Dogs bark.
Cats meow.
Dog barks and cat meows.
Cat meows and dog barks.

Dog barks.
Cat meows.
Dog barks.
Cat meows.
Dog barks loudly.
Cat meows softly.
A dog barks.
A cat meows.
The dog barks.
The cat meows.
Dogs bark.
Cats meow.
Dog barks and cat meows.
Cat meows and dog barks.

Dog barks.
Cat meows.
Dog barks.
Cat meows.
Dog barks loudly.
Cat meows softly.
A dog barks.
A cat meows.
The dog barks.
The cat meows.
Dogs bark.
Cats meow.
Dog barks and cat meows.
Cat meows and dog barks.
"""


# ============================================================
# 3. Character Tokenizer
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
# 4. Train / Validation Split
# ============================================================

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]


# ============================================================
# 5. Batch Generation
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
# 6. Self-Attention Head
# ============================================================

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        # Causal mask:
        # a character can only attend to itself
        # and characters that came before it.
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5)

        # Prevent looking into the future
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(
            wei,
            dim=-1
        )

        wei = self.dropout(wei)

        v = self.value(x)

        out = wei @ v

        return out


# ============================================================
# 7. Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_size
    ):

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

            nn.Linear(
                n_embd,
                4 * n_embd
            ),

            nn.GELU(),

            nn.Linear(
                4 * n_embd,
                n_embd
            ),

            nn.Dropout(dropout)
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# 9. Transformer Block
# ============================================================

class Block(nn.Module):

    def __init__(
        self,
        n_embd,
        n_head
    ):

        super().__init__()

        head_size = n_embd // n_head

        self.attention = MultiHeadAttention(
            n_head,
            head_size
        )

        self.feed_forward = FeedForward(
            n_embd
        )

        self.ln1 = nn.LayerNorm(
            n_embd
        )

        self.ln2 = nn.LayerNorm(
            n_embd
        )


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
# 10. MiniGPT Model
# ============================================================

class MiniGPT(nn.Module):

    def __init__(self):

        super().__init__()

        # Character embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        # Position embedding
        self.position_embedding = nn.Embedding(
            block_size,
            n_embd
        )

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    n_head
                )
                for _ in range(n_layer)
            ]
        )

        self.layer_norm = nn.LayerNorm(
            n_embd
        )

        # Convert hidden representation
        # back into vocabulary logits.
        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )


    def forward(
        self,
        idx,
        targets=None
    ):

        B, T = idx.shape

        # Character embeddings
        token_emb = self.token_embedding(idx)

        # Position embeddings
        positions = torch.arange(
            T,
            device=device
        )

        pos_emb = self.position_embedding(
            positions
        )

        # Combine character + position
        x = token_emb + pos_emb

        # Transformer
        x = self.blocks(x)

        x = self.layer_norm(x)

        # Vocabulary prediction
        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(
                B * T,
                C
            )

            targets = targets.view(
                B * T
            )

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss


    # ========================================================
    # Normal generation
    # ========================================================

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens
    ):

        self.eval()

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits, _ = self(
                idx_cond
            )

            logits = logits[:, -1, :]

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

        return idx


# ============================================================
# 11. Independent Content Guardrail
# ============================================================

class ExecutionGuardrail:

    def __init__(self):

        self.blocked_patterns = [

            r"rm\s+-rf",

            r"<script>",

            r"eval\(",

            r"Authorization:"
        ]


    def verify_payload(
        self,
        generated_text
    ):

        normalized_text = generated_text.strip()

        for pattern in self.blocked_patterns:

            if re.search(
                pattern,
                normalized_text,
                re.IGNORECASE
            ):

                print(
                    "SECURITY ALERT:",
                    "Guardrail tripped by pattern:",
                    pattern
                )

                return False

        return True


# ============================================================
# 12. Training and Experiment
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("Initializing MiniGPT")
    print("========================================")

    model = MiniGPT().to(device)

    print("\nModel parameters:",
          sum(p.numel() for p in model.parameters()))

    print("\n========================================")
    print("Training")
    print("========================================")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    model.train()

    for iteration in range(max_iters):

        xb, yb = get_batch("train")

        logits, loss = model(
            xb,
            yb
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        if iteration % 500 == 0:

            print(
                f"Iteration {iteration:4d} "
                f"| Loss {loss.item():.4f}"
            )


    print("\nTraining complete!")


    # ========================================================
    # 13. Test Prompt: DOG
    # ========================================================

    prompt = "Dog"

    context_tokens = torch.tensor(
        [encode(prompt)],
        dtype=torch.long,
        device=device
    )

    print("\n========================================")
    print("TEST 1")
    print("========================================")

    print("Prompt:", repr(prompt))

    output_tokens = model.generate(
        context_tokens,
        max_new_tokens=12
    )

    decoded_output = decode(
        output_tokens[0].tolist()
    )

    print("Generated:", repr(decoded_output))


    # ========================================================
    # 14. Test Prompt: CAT
    # ========================================================

    prompt = "Cat"

    context_tokens = torch.tensor(
        [encode(prompt)],
        dtype=torch.long,
        device=device
    )

    print("\n========================================")
    print("TEST 2")
    print("========================================")

    print("Prompt:", repr(prompt))

    output_tokens = model.generate(
        context_tokens,
        max_new_tokens=12
    )

    decoded_output = decode(
        output_tokens[0].tolist()
    )

    print("Generated:", repr(decoded_output))


    # ========================================================
    # 15. Test the Guardrail
    # ========================================================

    guard = ExecutionGuardrail()

    print("\n========================================")
    print("GUARDRAIL TEST")
    print("========================================")

    test_text = "Dog barks."

    print("Testing:", repr(test_text))

    if guard.verify_payload(test_text):

        print(
            "Layer 2 Passed: "
            "Content meets safety specifications."
        )

    else:

        print(
            "CRITICAL VETO: "
            "Content blocked by firewall."
        )


    # ========================================================
    # 16. Direct Comparison
    # ========================================================

    print("\n========================================")
    print("EXPERIMENT SUMMARY")
    print("========================================")

    print("""
The model was trained on examples such as:

    Dog barks.
    Cat meows.

The model then receives:

    Dog

and

    Cat

and predicts additional characters.

This is character-level next-token prediction.
""")

