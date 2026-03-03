# 2-class-neural-network-from-scratch-in-NumPy
Implemented forward/backward propagation, achieving 90% accuracy vs 47% baseline"

# Neural Network from Scratch — Planar Data Classification

A 2-class neural network built from scratch using **NumPy only** (no TensorFlow, no PyTorch).  
The model learns a non-linear decision boundary to classify a 2D flower-shaped dataset that logistic regression cannot solve.

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression (baseline) | ~47% |
| Neural Network (1 hidden layer) | ~90% |

---

## Architecture

```
Input (2) → Hidden Layer (4, tanh) → Output (1, sigmoid)
```

---

## Installation

```bash
pip install numpy matplotlib scikit-learn
```

---

## Run

```bash
python neural_network.py
```

---

## Project Structure

```
├── neural_network.py       # Full implementation
├── nn_flowchart.html       # Visual stage-by-stage flowchart
└── README.md
```

---

## Course

Deep Learning Specialization by Andrew Ng — Course 1, Week 3 (Coursera).  
All implementation is my own work following the assignment instructions.
