# mad-grad

A character-level Recurrent Neural Network built from scratch using only NumPy! no PyTorch, no TensorFlow, no autograd. Every forward pass, every gradient, every weight update written by hand.

Trained on the TinyShakespeare dataset to generate Shakespeare-style text.

---

## What this is

Most people learn deep learning by calling `.backward()` and trusting the framework. This project does the opposite, it derives and implements every equation manually, including:

- The RNN forward pass
- Backpropagation Through Time (BPTT)
- Numerical gradient checking to verify correctness
- Gradient clipping to handle exploding gradients
- SGD training loop with hidden state carry-over across sequences
- Character-level text generation with temperature sampling

The name **mad-grad** comes from manually computing gradients; the part of deep learning most people never see.

---

## Generated output

After 20 epochs with `hidden_size=256`, sample output at temperature 0.8:

```
ALONLO:
What is the country die it,
My daughter is love, whom you he make this ever.
```

The model kinda learns English word structure, punctuation rhythm, line breaks, and character name formatting purely from character-level patterns, no word embeddings, no tokenizer, no attention.

---

## How it works

### Architecture

A vanilla RNN with the following equations at each timestep `t`:

```
h_t = tanh(W_xh · x_t + W_hh · h_(t-1) + b_h)
y_t = W_hy · h_t + b_y
p_t = softmax(y_t)
```

Where:

- `x_t`: one-hot encoding of the input character
- `h_t`: hidden state (memory of the sequence so far)
- `p_t`: probability distribution over all characters

Loss is average cross-entropy over the sequence:

```
L = -1/T · Σ log(p_t[target_t])
```

### Backpropagation Through Time

Gradients flow backwards through every timestep. The key equations:

```
dlogits[t] = (p_t - one_hot(target_t)) / T
dW_hy      = dlogits.T @ h_states
dh_next    = W_hh.T @ da               # gradient passed to previous timestep
da         = dh_total * (1 - h_t²)     # tanh derivative
```

### Gradient checking

Before any training, every gradient is verified numerically using the centered difference formula:

```
∂L/∂θ ≈ (L(θ + ε) - L(θ - ε)) / 2ε
```

Relative error between analytical and numerical gradients is consistently below `1e-5`.

---

## Project structure

```
mad-grad/
├── rnn.py          # Full implementation — model, training loop, sampling
├── input.txt       # TinyShakespeare dataset (~1MB)
├── weights.npz     # Saved model weights (generated after first training run)
├── losses.npy      # Per-epoch average losses (generated after first training run)
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/thevoidop/mad-grad.git
cd mad-grad
pip install numpy
```

### 2. Get the dataset

```bash
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### 3. Train

```bash
python rnn.py
```

First run trains for 20 epochs and saves weights to `weights.npz`. Every subsequent run loads the saved weights and goes straight to sampling.

### 4. Sample

Sampling runs automatically after training. To change the seed or temperature, edit the bottom of `rnn.py`:

```python
print(rnn.sample("HAMLET: ", n=500, temperature=0.8))
```

---

## Hyperparameters

| Parameter     | Value | Notes                            |
| ------------- | ----- | -------------------------------- |
| `SEQ_LENGTH`  | 25    | Characters per training sequence |
| `HIDDEN_SIZE` | 256   | RNN hidden state dimension       |
| `EPOCHS`      | 20    | Full passes over the dataset     |
| `LR`          | 0.01  | SGD learning rate                |
| `max_norm`    | 5.0   | Gradient clipping threshold      |

---

## Training curve

Loss drops from ~4.17 (random) to below 1.6 over 20 epochs:

| Epoch | Avg Loss |
| ----- | -------- |
| 1     | ~2.78    |
| 5     | ~1.88    |
| 10    | ~1.65    |
| 20    | ~1.45    |

The expected loss at initialization is `ln(vocab_size) ≈ 4.17`. The model starts with no information and learns structure purely from gradient descent.

---

## Gradient checking results

Verified on all five parameter tensors before training:

```
W_xh(0, 0): analytical=-0.000123, numerical=-0.000123, rel_error=1.17e-07
W_hh(0, 1): analytical=0.000032,  numerical=0.000032,  rel_error=1.24e-06
W_hy(0, 0): analytical=-0.000731, numerical=-0.000731, rel_error=4.92e-08
b_h(0,):    analytical=-0.001666, numerical=-0.001666, rel_error=2.57e-09
b_y(0,):    analytical=-0.024624, numerical=-0.024624, rel_error=7.55e-10
```

All relative errors below `1e-5`. BPTT implementation is correct.

---

## What I learned building this

- How gradients actually flow through a recurrent network, not as a black box but as explicit matrix operations at every timestep
- Why the indexing of hidden states matters: `hs[t]` as `h_prev` and `hs[t+1]` as the post-activation state
- Why gradient clipping is necessary? Without it, a single bad sequence can send weights to infinity
- How temperature controls the sharpness of the probability distribution at generation time
- Why gradient checking is non-negotiable? It caught subtle bugs that would have been invisible from loss curves alone

---

## Possible extensions

- **AdaGrad / RMSProp**: adaptive learning rates improve convergence significantly over vanilla SGD
- **LSTM / GRU**: gated cells handle long-range dependencies much better than vanilla RNN
- **Mini-batch training**: process multiple sequences in parallel for faster training
- **Word-level model**: replace character one-hots with learned word embeddings
- **Truncated BPTT**: detach gradients after fixed steps for very long sequences

---

## References

- Karpathy, A. — [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- Karpathy, A. — [min-char-rnn](https://gist.github.com/karpathy/d4dee566867f8291f086) (the 112-line reference implementation this project expands on)
- Goodfellow, Bengio, Courville — _Deep Learning_, Chapter 10 (Sequence Modeling)

---

## Dependencies

```
numpy
```

That's it.
