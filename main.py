import torch
import torch.nn as nn
import torch.nn.functional as F
import re


# ============================================================
# 1. CONFIGURATION
# ============================================================

batch_size = 32
block_size = 128

max_iters = 3000
learning_rate = 3e-4

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1

semantic_epochs = 2000
semantic_learning_rate = 0.01

## can I use cuda soft stack
device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)


print("=" * 60)
print("ANIMAL CONSTRAINT LEARNING EXPERIMENT")
print("=" * 60)

print("PyTorch version:", torch.__version__)
print("Tensor test:", torch.tensor([1, 2, 3]))
print("Using device:", device)


# ============================================================
# 2. ANIMAL KNOWLEDGE
# ============================================================
#
# This is our small "ground truth" knowledge base.
#
# The model should learn relationships such as:
#
#     dog      -> barks
#     cat      -> meows
#     cow      -> moos
#
# and should NOT learn:
#
#     cow      -> barks
#     dog      -> moos
#
# ============================================================

ANIMAL_SOUNDS = {

    # Farm Animals

    "dog": {
        "verb": "barks",
        "sound": "woof woof"
    },

    "cat": {
        "verb": "meows",
        "sound": "meow meow"
    },

    "cow": {
        "verb": "moos",
        "sound": "moo"
    },

    "sheep": {
        "verb": "bleats",
        "sound": "baa baa"
    },

    "goat": {
        "verb": "bleats",
        "sound": "baa baa"
    },

    "pig": {
        "verb": "oinks",
        "sound": "oink oink"
    },

    "horse": {
        "verb": "neighs",
        "sound": "neigh"
    },

    "donkey": {
        "verb": "brays",
        "sound": "hee-haw"
    },


    # Birds and Small Wildlife

    "bird": {
        "verb": "chirps",
        "sound": "chirp chirp"
    },

    "duck": {
        "verb": "quacks",
        "sound": "quack quack"
    },

    "chicken": {
        "verb": "clucks",
        "sound": "cluck cluck"
    },

    "rooster": {
        "verb": "crows",
        "sound": "cock-a-doodle-doo"
    },

    "crow": {
        "verb": "caws",
        "sound": "caw caw"
    },

    "owl": {
        "verb": "hoots",
        "sound": "hoo hoo"
    },

    "mouse": {
        "verb": "squeaks",
        "sound": "squeak squeak"
    },

    "frog": {
        "verb": "croaks",
        "sound": "ribbit ribbit"
    },


    # Wild and Exotic Animals

    "lion": {
        "verb": "roars",
        "sound": "roar"
    },

    "tiger": {
        "verb": "roars",
        "sound": "roar"
    },

    "wolf": {
        "verb": "howls",
        "sound": "ahoo"
    },

    "elephant": {
        "verb": "trumpets",
        "sound": "pawoooo"
    },

    "snake": {
        "verb": "hisses",
        "sound": "sssss"
    },

    "bee": {
        "verb": "buzzes",
        "sound": "buzz"
    },

    "monkey": {
        "verb": "chatters",
        "sound": "ooh-ooh aah-aah"
    }
}


# ============================================================
# 3. CREATE INTEGER CLASS IDs
# ============================================================
#
# These numbers are identifiers.
#
# They are NOT learned weights.
#
# Example:
#
#     dog  -> 0
#     cat  -> 1
#     cow  -> 2
#     sheep -> 3
#
# Therefore:
#
#     dog, cow, cat, sheep
#
# becomes:
#
#     [0, 2, 1, 3]
#
# ============================================================

animals = list(ANIMAL_SOUNDS.keys())

animal_to_id = {
    animal: i
    for i, animal in enumerate(animals)
}

id_to_animal = {
    i: animal
    for animal, i in animal_to_id.items()
}


print("\n" + "=" * 60)
print("ANIMAL CLASS IDs")
print("=" * 60)

for animal, animal_id in animal_to_id.items():

    print(
        f"{animal_id:2d} -> {animal}"
    )


# ============================================================
# 4. CREATE VERB CLASS IDs
# ============================================================

verbs = sorted(
    set(
        info["verb"]
        for info in ANIMAL_SOUNDS.values()
    )
)

verb_to_id = {
    verb: i
    for i, verb in enumerate(verbs)
}

id_to_verb = {
    i: verb
    for verb, i in verb_to_id.items()
}


print("\n" + "=" * 60)
print("VERB CLASS IDs")
print("=" * 60)

for verb, verb_id in verb_to_id.items():

    print(
        f"{verb_id:2d} -> {verb}"
    )


# ============================================================
# 5. DEMONSTRATE [0, 2, 1, 3]
# ============================================================

test_animals = [
    "dog",
    "cow",
    "cat",
    "sheep"
]

test_ids = [
    animal_to_id[a]
    for a in test_animals
]


print("\n" + "=" * 60)
print("INTEGER ENCODING TEST")
print("=" * 60)

print("Animals:")
print(test_animals)

print("\nIDs:")
print(test_ids)

print("\nMapping:")

for animal, animal_id in zip(
    test_animals,
    test_ids
):

    print(
        f"{animal_id} -> {animal}"
    )


# ============================================================
# 6. BUILD CHARACTER-LEVEL TRAINING TEXT
# ============================================================
#
# This is the language-model portion.
#
# We generate many natural-language examples from our
# knowledge table.
#
# ============================================================

training_sentences = []

for animal, info in ANIMAL_SOUNDS.items():

    capitalized = animal.capitalize()

    verb = info["verb"]

    sound = info["sound"]

    training_sentences.extend([

        f"{capitalized} {verb}.\n",

        f"The {animal} {verb}.\n",

        f"A {animal} {verb}.\n",

        f"{capitalized} {verb} {sound}.\n",

        f"What does a {animal} do? "
        f"{capitalized} {verb}.\n",

    ])


# Repeat because this is a very small corpus.

text = "".join(
    training_sentences * 20
)


# ============================================================
# 7. CHARACTER TOKENIZER
# ============================================================

chars = sorted(
    list(set(text))
)

vocab_size = len(chars)

stoi = {
    ch: i
    for i, ch in enumerate(chars)
}

itos = {
    i: ch
    for ch, i in stoi.items()
}


def encode(s):

    return [
        stoi[c]
        for c in s
        if c in stoi
    ]


def decode(ids):

    return "".join(
        itos[i]
        for i in ids
    )


data = torch.tensor(
    encode(text),
    dtype=torch.long
)


print("\n" + "=" * 60)
print("CHARACTER DATASET")
print("=" * 60)

print("Vocabulary size:", vocab_size)
print("Dataset size:", len(data))


# ============================================================
# 8. TRAIN / VALIDATION SPLIT
# ============================================================

n = int(
    0.9 * len(data)
)

train_data = data[:n]

val_data = data[n:]


# ============================================================
# 9. BATCH GENERATOR
# ============================================================

def get_batch(split):

    source = (
        train_data
        if split == "train"
        else val_data
    )

    max_start = len(source) - block_size - 1

    ix = torch.randint(
        max_start,
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

    return (
        x.to(device),
        y.to(device)
    )


# ============================================================
# 10. SELF-ATTENTION HEAD
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

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )


    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)

        q = self.query(x)

        wei = q @ k.transpose(
            -2,
            -1
        )

        wei = wei * (
            k.shape[-1] ** -0.5
        )

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(
            wei,
            dim=-1
        )

        wei = self.dropout(
            wei
        )

        v = self.value(x)

        out = wei @ v

        return out


# ============================================================
# 11. MULTI-HEAD ATTENTION
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

        self.dropout = nn.Dropout(
            dropout
        )


    def forward(self, x):

        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )

        out = self.projection(
            out
        )

        return self.dropout(
            out
        )


# ============================================================
# 12. FEED-FORWARD NETWORK
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

            nn.Dropout(
                dropout
            )
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# 13. TRANSFORMER BLOCK
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

        x = x + self.attention(
            self.ln1(x)
        )

        x = x + self.feed_forward(
            self.ln2(x)
        )

        return x


# ============================================================
# 14. MINIGPT
# ============================================================

class MiniGPT(nn.Module):

    def __init__(self):

        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding = nn.Embedding(
            block_size,
            n_embd
        )

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

        token_emb = self.token_embedding(
            idx
        )

        positions = torch.arange(
            T,
            device=device
        )

        pos_emb = self.position_embedding(
            positions
        )

        x = token_emb + pos_emb

        x = self.blocks(x)

        x = self.layer_norm(x)

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


    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens
    ):

        self.eval()

        for _ in range(
            max_new_tokens
        ):

            idx_cond = idx[
                :, -block_size:
            ]

            logits, _ = self(
                idx_cond
            )

            logits = logits[
                :, -1, :
            ]

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            idx = torch.cat(
                (
                    idx,
                    next_token
                ),
                dim=1
            )

        return idx


# ============================================================
# 15. SEMANTIC MODEL
# ============================================================
#
# This model explicitly learns:
#
#       animal ID -> correct verb ID
#
# The embedding contains LEARNED PARAMETERS.
#
# ============================================================

class AnimalSemanticModel(nn.Module):

    def __init__(
        self,
        num_animals,
        num_verbs,
        embedding_dim=32
    ):

        super().__init__()

        self.animal_embedding = nn.Embedding(
            num_animals,
            embedding_dim
        )

        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                num_verbs
            )
        )


    def forward(
        self,
        animal_ids
    ):

        x = self.animal_embedding(
            animal_ids
        )

        logits = self.network(
            x
        )

        return logits


# ============================================================
# 16. TRAIN SEMANTIC MODEL
# ============================================================

def train_semantic_model():

    print("\n" + "=" * 60)
    print("TRAINING SEMANTIC MODEL")
    print("=" * 60)

    model = AnimalSemanticModel(
        num_animals=len(animals),
        num_verbs=len(verbs)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=semantic_learning_rate
    )

    animal_ids = torch.tensor(
        [
            animal_to_id[a]
            for a in animals
        ],
        dtype=torch.long,
        device=device
    )

    target_verb_ids = torch.tensor(
        [
            verb_to_id[
                ANIMAL_SOUNDS[a]["verb"]
            ]

            for a in animals
        ],
        dtype=torch.long,
        device=device
    )

    model.train()

    for epoch in range(
        semantic_epochs
    ):

        logits = model(
            animal_ids
        )

        loss = F.cross_entropy(
            logits,
            target_verb_ids
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if epoch % 200 == 0:

            print(
                f"Epoch {epoch:4d} "
                f"| Loss {loss.item():.6f}"
            )

    return model


# ============================================================
# 17. SEMANTIC PREDICTION
# ============================================================

@torch.no_grad()
def predict_animal(
    model,
    animal
):

    model.eval()

    animal_id = torch.tensor(
        [animal_to_id[animal]],
        dtype=torch.long,
        device=device
    )

    logits = model(
        animal_id
    )

    probabilities = F.softmax(
        logits,
        dim=-1
    )

    predicted_id = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    return (
        id_to_verb[predicted_id],
        probabilities[0]
    )


# ============================================================
# 18. SHOW EVERYTHING THE SEMANTIC MODEL LEARNED
# ============================================================

def show_predictions(
    model
):

    print("\n" + "=" * 60)
    print("SEMANTIC MODEL RESULTS")
    print("=" * 60)

    for animal in animals:

        predicted, probabilities = (
            predict_animal(
                model,
                animal
            )
        )

        expected = ANIMAL_SOUNDS[
            animal
        ]["verb"]

        probability = probabilities[
            verb_to_id[predicted]
        ].item()

        status = (
            "CORRECT"
            if predicted == expected
            else "WRONG"
        )

        print(
            f"{animal:10s} "
            f"expected={expected:10s} "
            f"predicted={predicted:10s} "
            f"P={probability:.4f} "
            f"{status}"
        )


# ============================================================
# 19. DELIBERATELY TEST WRONG RELATIONSHIPS
# ============================================================
#
# This is the key experiment.
#
# We ask:
#
#       What probability does the trained model assign
#       to an impossible relationship?
#
# ============================================================

def test_wrong_relationships(
    model
):

    tests = [

        ("cow", "barks"),
        ("cow", "meows"),
        ("cow", "neighs"),

        ("dog", "moos"),
        ("dog", "meows"),

        ("cat", "barks"),
        ("cat", "oinks"),

        ("horse", "barks"),
        ("horse", "meows"),

        ("lion", "quacks"),
        ("lion", "barks"),

        ("elephant", "barks"),
        ("elephant", "meows"),

        ("snake", "barks"),
        ("snake", "meows"),

        ("frog", "barks"),
        ("frog", "roars")
    ]

    print("\n" + "=" * 60)
    print("WRONG-RELATIONSHIP TEST")
    print("=" * 60)

    for animal, wrong_verb in tests:

        predicted, probabilities = (
            predict_animal(
                model,
                animal
            )
        )

        correct_verb = ANIMAL_SOUNDS[
            animal
        ]["verb"]

        correct_probability = probabilities[
            verb_to_id[correct_verb]
        ].item()

        wrong_probability = probabilities[
            verb_to_id[wrong_verb]
        ].item()

        print(
            f"{animal:10s} "
            f"correct={correct_verb:10s} "
            f"P={correct_probability:.6f} | "
            f"WRONG {wrong_verb:10s} "
            f"P={wrong_probability:.6f}"
        )


# ============================================================
# 20. [0,2,1,3] VECTOR TEST
# ============================================================

def test_id_vector(
    model
):

    print("\n" + "=" * 60)
    print("[0,2,1,3] VECTOR TEST")
    print("=" * 60)

    test_animals = [
        "dog",
        "cow",
        "cat",
        "sheep"
    ]

    ids = torch.tensor(
        [
            animal_to_id[a]
            for a in test_animals
        ],
        dtype=torch.long,
        device=device
    )

    print(
        "Input:",
        test_animals
    )

    print(
        "IDs:",
        ids.tolist()
    )

    logits = model(
        ids
    )

    probabilities = F.softmax(
        logits,
        dim=-1
    )

    predicted_ids = torch.argmax(
        probabilities,
        dim=1
    )

    print("\nOutput:")

    for animal, verb_id in zip(
        test_animals,
        predicted_ids.tolist()
    ):

        print(
            f"{animal:10s} -> "
            f"{id_to_verb[verb_id]}"
        )


# ============================================================
# 21. LEARNED MODEL VS GUARDRAIL
# ============================================================
#
# The learned model predicts what SHOULD happen.
#
# The guardrail independently checks what was generated.
#
# ============================================================

class AnimalGuardrail:

    def __init__(
        self,
        animal_sounds
    ):

        self.animal_sounds = animal_sounds


    def verify(
        self,
        animal,
        verb
    ):

        if animal not in self.animal_sounds:

            print(
                "GUARDRAIL: Unknown animal."
            )

            return False

        correct_verb = self.animal_sounds[
            animal
        ]["verb"]

        if verb != correct_verb:

            print(
                "GUARDRAIL BLOCKED:"
            )

            print(
                f"    Animal : {animal}"
            )

            print(
                f"    Verb   : {verb}"
            )

            print(
                f"    Expected: {correct_verb}"
            )

            return False

        print(
            "GUARDRAIL PASSED:"
        )

        print(
            f"    {animal} -> {verb}"
        )

        return True


# ============================================================
# 22. ORIGINAL SECURITY GUARDRAIL
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

        normalized_text = (
            generated_text.strip()
        )

        for pattern in (
            self.blocked_patterns
        ):

            if re.search(
                pattern,
                normalized_text,
                re.IGNORECASE
            ):

                print(
                    "SECURITY ALERT:",
                    pattern
                )

                return False

        return True


# ============================================================
# 23. TRAIN MINIGPT
# ============================================================

def train_minigpt():

    print("\n" + "=" * 60)
    print("TRAINING MINIGPT")
    print("=" * 60)

    model = MiniGPT().to(device)

    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    model.train()

    for iteration in range(
        max_iters
    ):

        xb, yb = get_batch(
            "train"
        )

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

    return model


# ============================================================
# 24. MINIGPT GENERATION
# ============================================================

def generate_text(
    model,
    prompt,
    max_new_tokens=30
):

    encoded = encode(
        prompt
    )

    context = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device
    )

    output = model.generate(
        context,
        max_new_tokens
    )

    return decode(
        output[0].tolist()
    )


# ============================================================
# 25. TEST MINIGPT
# ============================================================

def test_minigpt(
    model
):

    print("\n" + "=" * 60)
    print("MINIGPT GENERATION TEST")
    print("=" * 60)

    prompts = [

        "Dog",

        "Cat",

        "Cow",

        "Sheep",

        "Horse",

        "Lion",

        "Elephant",

        "Snake",

        "What does a cow do? ",

        "What does a dog do? "

    ]

    for prompt in prompts:

        result = generate_text(
            model,
            prompt,
            30
        )

        print(
            "\nPrompt:",
            repr(prompt)
        )

        print(
            "Generated:",
            repr(result)
        )


# ============================================================
# 26. TEST LEARNED MODEL + GUARDRAIL
# ============================================================

def test_guardrail(
    semantic_model
):

    print("\n" + "=" * 60)
    print("LEARNED MODEL + GUARDRAIL TEST")
    print("=" * 60)

    guardrail = AnimalGuardrail(
        ANIMAL_SOUNDS
    )

    tests = [

        ("cow", "moos"),

        ("cow", "barks"),

        ("dog", "barks"),

        ("dog", "moos"),

        ("cat", "meows"),

        ("cat", "barks"),

        ("horse", "neighs"),

        ("horse", "barks")
    ]

    for animal, proposed_verb in tests:

        print("\nTesting:")
        print(
            f"    {animal} -> {proposed_verb}"
        )

        guardrail.verify(
            animal,
            proposed_verb
        )


# ============================================================
# 27. MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # STEP 1
    # Create and display the integer representation.
    # --------------------------------------------------------

    print("\nSTEP 1: INTEGER REPRESENTATION")

    print(
        f"{test_animals} -> {test_ids}"
    )


    # --------------------------------------------------------
    # STEP 2
    # Train explicit semantic model.
    # --------------------------------------------------------

    semantic_model = train_semantic_model()


    # --------------------------------------------------------
    # STEP 3
    # See what semantic model learned.
    # --------------------------------------------------------

    show_predictions(
        semantic_model
    )


    # --------------------------------------------------------
    # STEP 4
    # Test impossible relationships.
    # --------------------------------------------------------

    test_wrong_relationships(
        semantic_model
    )


    # --------------------------------------------------------
    # STEP 5
    # Test [0,2,1,3].
    # --------------------------------------------------------

    test_id_vector(
        semantic_model
    )


    # --------------------------------------------------------
    # STEP 6
    # Train the original character-level MiniGPT.
    # --------------------------------------------------------

    minigpt = train_minigpt()


    # --------------------------------------------------------
    # STEP 7
    # Generate natural language.
    # --------------------------------------------------------

    test_minigpt(
        minigpt
    )


    # --------------------------------------------------------
    # STEP 8
    # Test learned semantic model against guardrail.
    # --------------------------------------------------------

    test_guardrail(
        semantic_model
    )


    # --------------------------------------------------------
    # STEP 9
    # Original security guardrail test.
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ORIGINAL SECURITY GUARDRAIL TEST")
    print("=" * 60)

    execution_guardrail = ExecutionGuardrail()

    test_text = "Dog barks."

    print(
        "Testing:",
        repr(test_text)
    )

    if execution_guardrail.verify_payload(
        test_text
    ):

        print(
            "Layer 2 Passed."
        )

    else:

        print(
            "CRITICAL VETO."
        )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL EXPERIMENT SUMMARY")
    print("=" * 60)

    print(
        """
1. ANIMAL DATA

   ANIMAL_SOUNDS defines the desired relationships.

2. INTEGER IDs

   Animals are converted into IDs.

   Example:

       dog, cow, cat, sheep
          ->
       [0, 2, 1, 3]

   These are identifiers, NOT weights.

3. SEMANTIC TRAINING

   The semantic neural network learns:

       animal ID -> verb ID

   using cross-entropy loss.

4. WRONG RELATIONSHIPS

   We explicitly test:

       cow -> barks
       dog -> moos
       cat -> barks

   and examine their probabilities.

5. MINIGPT

   MiniGPT independently learns character-level language
   patterns such as:

       Dog barks.
       Cat meows.
       Cow moos.

6. LEARNED MODEL

   The semantic model is trained to prefer the correct
   animal/action relationship.

7. GUARDRAIL

   The AnimalGuardrail independently rejects an incorrect
   relationship.

8. SECURITY GUARDRAIL

   The original ExecutionGuardrail remains separate and
   protects against patterns such as:

       rm -rf
       <script>
       eval(...)
       Authorization:

IMPORTANT:

The semantic model and MiniGPT are currently separate.

The next experiment can combine them into ONE Transformer
training objective:

       total_loss =
           language_loss
           +
           lambda * semantic_loss

so that the MiniGPT itself learns the animal constraint
through backpropagation.
"""
    )
