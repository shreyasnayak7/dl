"""
Streamlit Web Interface for Visualizing Layer Freezing in a CNN

Run using:
    streamlit run layer_freeze_web_app.py

Features:
- Load a simple CNN
- UI checkboxes to freeze/unfreeze layers
- Train on random data for a few steps
- Visualize which layers changed (bar chart + table)

Requirements:
    pip install streamlit torch matplotlib
"""
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import OrderedDict

# ------------------------------
# Model Definition
# ------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(1, 8, 3, padding=1)),
            ("relu1", nn.ReLU()),
            ("pool1", nn.MaxPool2d(2)),
            ("conv2", nn.Conv2d(8, 16, 3, padding=1)),
            ("relu2", nn.ReLU()),
            ("pool2", nn.MaxPool2d(2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(16*7*7, 64)),
            ("relu3", nn.ReLU()),
            ("fc2", nn.Linear(64, 10)),
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ------------------------------
# Helper functions
# ------------------------------
def freeze_layers(model, layer_names):
    for name, param in model.named_parameters():
        for layer in layer_names:
            if layer in name:
                param.requires_grad = False


def snapshot(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def compute_diff(before, after):
    diffs = {}
    for k in before:
        diff = (after[k] - before[k]).abs().max().item()
        module = k.split('.')[0]
        diffs.setdefault(module, []).append(diff)
    return {m: max(v) for m, v in diffs.items()}


def train_steps(model, steps=20, lr=1e-2, batch=32):
    opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        x = torch.randn(batch, 1, 28, 28)
        y = torch.randint(0, 10, (batch,))
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🔐 Layer Freezing Visualizer for CNN (PyTorch)")
st.write("Select layers to freeze, train the model briefly, and visualize which layers learned.")

# Initialize model
model = SimpleCNN()
layer_options = ["conv1", "conv2", "fc1", "fc2"]

st.sidebar.header("Freeze Layers")
freeze_selected = []
for layer in layer_options:
    if st.sidebar.checkbox(f"Freeze {layer}"):
        freeze_selected.append(layer)

steps = st.sidebar.slider("Training Steps", min_value=5, max_value=100, value=20)
lr = st.sidebar.number_input("Learning Rate", value=0.01)
batch = st.sidebar.number_input("Batch Size", value=32)

if st.button("Run Training and Visualize"):
    st.subheader("Results")

    # Apply freezing
    freeze_layers(model, freeze_selected)

    # Take snapshot
    before = snapshot(model)

    # Train
    train_steps(model, steps=steps, lr=lr, batch=batch)

    # After snapshot
    after = snapshot(model)

    # Compute diffs
    diffs = compute_diff(before, after)

    # Display table
    st.write("### Layer Parameter Changes (Max Abs Diff)")
    st.table({"Layer": list(diffs.keys()), "Max Change": list(diffs.values())})

    # Bar plot
    fig, ax = plt.subplots()
    ax.bar(list(diffs.keys()), list(diffs.values()))
    ax.set_ylabel("Max Param Change")
    ax.set_title("Layer Learning Visualization")
    st.pyplot(fig)

st.info("Use the checkboxes on the left to freeze specific layers.")