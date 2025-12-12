import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import re
from collections import Counter

st.set_page_config(page_title="Word2Vec Generator", layout="wide")
st.title("🧠 Neural Network Word2Vec Embedding Generator")

st.write("Enter corpus text below, adjust parameters, and train a simple Word2Vec model.")

# ------------------------------
# User Inputs
# ------------------------------
corpus = st.text_area("Enter Text Corpus:", 
"""Natural language processing enables computers to understand human language.
Word embeddings map words into continuous vector space.
Neural networks can be used to learn these embeddings from context words in a document corpus."""
)

window_size = st.slider("Context Window Size", 1, 5, 2)
embed_dim = st.slider("Embedding Dimension", 10, 100, 50)
epochs = st.slider("Training Epochs", 10, 200, 50)

# ------------------------------
# Tokenizer
# ------------------------------
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# ------------------------------
# Train Button
# ------------------------------
if st.button("Train Word2Vec Model"):

    tokens = tokenize(corpus)

    if len(tokens) < 5:
        st.error("Please enter more text!")
        st.stop()

    vocab = {w: i for i, w in enumerate(Counter(tokens))}
    id2word = {i: w for w, i in vocab.items()}

    # ----------------------------------
    # Generate Skip-gram training pairs
    # ----------------------------------
    def generate_pairs(tokens, window=2):
        pairs = []
        for i, w in enumerate(tokens):
            for j in range(max(0, i-window), min(len(tokens), i+window+1)):
                if i != j:
                    pairs.append((vocab[w], vocab[tokens[j]]))
        return pairs

    training_data = generate_pairs(tokens, window_size)

    # ----------------------------------
    # Word2Vec Model
    # ----------------------------------
    class Word2Vec(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, embed_dim)
            self.out = nn.Linear(embed_dim, vocab_size)

        def forward(self, x):
            return self.out(self.emb(x))

    model = Word2Vec(len(vocab), embed_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    st.write("### 🔄 Training...")
    progress = st.progress(0)

    for epoch in range(epochs):
        total_loss = 0
        for c, o in training_data:
            c = torch.tensor([c])
            o = torch.tensor([o])

            optimizer.zero_grad()
            loss = criterion(model(c), o)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        progress.progress((epoch + 1) / epochs)
        if (epoch + 1) % 10 == 0:
            st.write(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    st.success("Training Completed ✔")

    # Save to session_state
    st.session_state["vocab"] = vocab
    st.session_state["embeddings"] = model.emb.weight.data
    st.session_state["trained"] = True


# ------------------------------
# Embedding Lookup Section
# ------------------------------
st.subheader("🔍 Lookup Word Embedding")

query_word = st.text_input("Enter a word to view its embedding:")

if st.button("Get Embedding"):
    if "trained" not in st.session_state:
        st.error("⚠ Please train the model first!")
    else:
        vocab = st.session_state["vocab"]
        embeddings = st.session_state["embeddings"]

        if query_word.lower() in vocab:
            st.write(f"### Embedding for **{query_word}**")
            st.write(embeddings[vocab[query_word.lower()]])
        else:
            st.error("Word not found in vocabulary!")
