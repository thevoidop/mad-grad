import numpy as np
import os

# ─── Hyperparameters ──────────────────────────────────────────────────────────
SEQ_LENGTH = 25
HIDDEN_SIZE = 256
EPOCHS = 20
LR = 0.01
WEIGHTS = "weights.npz"
LOSSES = "losses.npy"

# ─── Data loading ─────────────────────────────────────────────────────────────
with open("input.txt", "r", encoding="utf-8") as f:
    content = f.read()

chars = sorted(set(content))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
data = [char_to_idx[ch] for ch in content]

# Slice into non-overlapping (input, target) sequence pairs
# target is input shifted one step to the right
inputs_all, targets_all = [], []
for i in range(0, len(data) - SEQ_LENGTH - 1, SEQ_LENGTH):
    inputs_all.append(data[i : i + SEQ_LENGTH])
    targets_all.append(data[i + 1 : i + SEQ_LENGTH + 1])

print(f"vocab_size: {vocab_size}  |  sequences: {len(inputs_all)}")


# ─── Model ────────────────────────────────────────────────────────────────────
class RNN:
    def __init__(self, hidden_size, vocab_size):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Weight matrices — small random init to break symmetry
        self.W_xh = np.random.randn(hidden_size, vocab_size) * 0.01  # input → hidden
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden → hidden
        self.W_hy = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden → output
        self.b_h = np.zeros(hidden_size)  # hidden bias
        self.b_y = np.zeros(vocab_size)  # output bias

    # ── Helpers ───────────────────────────────────────────────────────────────

    def one_hot(self, idx):
        """Convert an integer index to a one-hot vector of length vocab_size."""
        x = np.zeros(self.vocab_size)
        x[idx] = 1
        return x

    def softmax(self, x):
        """Numerically stable softmax — subtract max before exponentiation."""
        e = np.exp(x - np.max(x))
        return e / e.sum()

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, inputs, h_prev):
        """
        Run the sequence through the RNN one step at a time.

        Stores xs, hs, ps as instance attributes for use in backward().
          xs — one-hot inputs,         list[seq_length]
          hs — hidden states,          list[seq_length + 1]  (hs[0] = h_prev)
          ps — softmax probabilities,  list[seq_length]

        Returns xs, hs, ps.
        """
        self.xs = []
        self.hs = [h_prev]  # hs[0] is h_prev; hs[t+1] is the state after step t
        self.ps = []

        for idx in inputs:
            x_t = self.one_hot(idx)
            h_t = np.tanh(self.W_xh @ x_t + self.W_hh @ self.hs[-1] + self.b_h)
            y_t = self.W_hy @ h_t + self.b_y
            p_t = self.softmax(y_t)

            self.xs.append(x_t)
            self.hs.append(h_t)
            self.ps.append(p_t)

        return self.xs, self.hs, self.ps

    # ── Loss ──────────────────────────────────────────────────────────────────

    def loss(self, targets):
        """
        Average cross-entropy loss over the sequence.
        Expects forward() to have been called first.
        """
        total = 0.0
        for t, target in enumerate(targets):
            p = max(self.ps[t][target], 1e-12)  # clip to avoid log(0)
            total += -np.log(p)
        return total / len(targets)

    # ── Backward pass (BPTT) ──────────────────────────────────────────────────

    def backward(self, targets):
        """
        Backpropagation through time.
        Expects forward() to have been called first.

        Returns a dict of gradients for every parameter.
        """
        T = len(targets)

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        # ── Output layer ──────────────────────────────────────────────────────
        # Gradient of softmax + cross-entropy combined: p - one_hot(target)
        dlogits = np.array(self.ps)  # (T, vocab_size)
        for t, target in enumerate(targets):
            dlogits[t, target] -= 1
        dlogits /= T  # average over steps

        h_states = np.array(self.hs[1:])  # (T, hidden_size)
        dW_hy = dlogits.T @ h_states  # (vocab_size, hidden_size)
        db_y = dlogits.sum(axis=0)

        # Gradient of loss w.r.t. each hidden state from the output layer
        dh = dlogits @ self.W_hy  # (T, hidden_size)

        # ── Recurrent layer — loop backwards through time ─────────────────────
        dh_next = np.zeros(self.hidden_size)

        for t in range(T - 1, -1, -1):
            dh_total = dh[t] + dh_next  # gradient from output + future
            da = dh_total * (1 - self.hs[t + 1] ** 2)  # tanh derivative

            dW_xh += np.outer(da, self.xs[t])  # outer product: (hidden, vocab)
            dW_hh += np.outer(da, self.hs[t])  # hs[t] is h_prev for step t
            db_h += da
            dh_next = self.W_hh.T @ da  # pass gradient to previous step

        return {"W_xh": dW_xh, "W_hh": dW_hh, "W_hy": dW_hy, "b_h": db_h, "b_y": db_y}

    # ── Gradient clipping ─────────────────────────────────────────────────────

    def clip_gradients(self, grads, max_norm=5.0):
        """
        Scale all gradients down if their global L2 norm exceeds max_norm.
        Prevents exploding gradients during training.
        """
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        if total_norm > max_norm:
            scale = max_norm / total_norm
            grads = {k: v * scale for k, v in grads.items()}
        return grads

    # ── Parameter update (SGD) ────────────────────────────────────────────────

    def update_params(self, grads, lr):
        """Vanilla SGD: subtract learning-rate-scaled gradients in place."""
        for name in ["W_xh", "W_hh", "W_hy", "b_h", "b_y"]:
            getattr(self, name)[:] -= lr * grads[name]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path=WEIGHTS):
        """Save all weight matrices and biases to a .npz file."""
        np.savez(
            path,
            W_xh=self.W_xh,
            W_hh=self.W_hh,
            W_hy=self.W_hy,
            b_h=self.b_h,
            b_y=self.b_y,
        )
        print(f"Weights saved → {path}")

    def load(self, path=WEIGHTS):
        """Load weights from a previously saved .npz file."""
        d = np.load(path)
        self.W_xh, self.W_hh, self.W_hy = d["W_xh"], d["W_hh"], d["W_hy"]
        self.b_h, self.b_y = d["b_h"], d["b_y"]
        print(f"Weights loaded ← {path}")

    # ── Text generation ───────────────────────────────────────────────────────

    def sample(self, seed, n, temperature=1.0):
        """
        Generate n characters starting from a seed string.

        temperature < 1 → more predictable output
        temperature > 1 → more random/creative output
        """
        h = np.zeros(self.hidden_size)
        out = seed

        # Warm up the hidden state by feeding the seed characters
        for ch in seed:
            x = self.one_hot(char_to_idx[ch])
            h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)

        idx = char_to_idx[seed[-1]]

        for _ in range(n):
            x = self.one_hot(idx)
            h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)
            y = self.W_hy @ h + self.b_y
            p = self.softmax(y / temperature)  # temperature rescales logits
            idx = np.random.choice(len(p), p=p)
            out += idx_to_char[idx]

        return out


# ─── Training loop ────────────────────────────────────────────────────────────


def train(rnn, inputs_all, targets_all, epochs=EPOCHS, lr=LR):
    """
    Train the RNN for a given number of epochs.
    Carries the hidden state across consecutive sequences within each epoch.
    Returns a list of per-epoch average losses.
    """
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        h_prev = np.zeros(rnn.hidden_size)

        for i in range(len(inputs_all)):
            rnn.forward(inputs_all[i], h_prev)
            loss = rnn.loss(targets_all[i])
            grads = rnn.backward(targets_all[i])
            grads = rnn.clip_gradients(grads)
            rnn.update_params(grads, lr)

            h_prev = rnn.hs[-1]  # carry hidden state into next sequence
            total_loss += loss

            if i % 5000 == 0:
                print(f"  epoch {epoch+1}/{epochs}  seq {i}  loss {loss:.4f}")

        avg = total_loss / len(inputs_all)
        epoch_losses.append(avg)
        print(f"── epoch {epoch+1} complete  avg loss: {avg:.4f} ──")

        # Sample every 5 epochs so you can watch quality improve
        if (epoch + 1) % 5 == 0:
            print(rnn.sample("ROMEO: ", n=200, temperature=0.8))

    return epoch_losses


# ─── Main ─────────────────────────────────────────────────────────────────────

rnn = RNN(HIDDEN_SIZE, vocab_size)

if os.path.exists(WEIGHTS):
    rnn.load()
    losses = list(np.load(LOSSES)) if os.path.exists(LOSSES) else []
else:
    losses = train(rnn, inputs_all, targets_all)
    rnn.save()
    np.save(LOSSES, np.array(losses))

# ─── Samples ──────────────────────────────────────────────────────────────────

print("\n--- Temperature 0.5 ---")
print(rnn.sample("ROMEO: ", n=500, temperature=0.5))

print("\n--- Temperature 1.0 ---")
print(rnn.sample("ROMEO: ", n=500, temperature=1.0))

print("\n--- Temperature 1.5 ---")
print(rnn.sample("ROMEO: ", n=500, temperature=1.5))
