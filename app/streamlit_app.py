import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import json
import os

# --- 1. Custom Layer ---
@tf.keras.utils.register_keras_serializable()
class AttentionPooling1D(layers.Layer):
    """Learns a per-timestep importance score, normalizes it with softmax,
    and returns the weighted sum over the sequence dimension.
    Also exposes the attention weights for later visualization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(1)

    def call(self, inputs):
        # inputs shape: (batch, seq_len, channels)
        scores = self.score_dense(inputs)               # (batch, seq_len, 1)
        weights = tf.nn.softmax(scores, axis=1)          # (batch, seq_len, 1)
        weighted_sum = tf.reduce_sum(inputs * weights, axis=1)  # (batch, channels)
        return weighted_sum, weights

    def get_config(self):
        return super().get_config()

# --- 2. Encodings and Charsets ---
CHARISOSMISET = {"#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, "/": 7,
    ".": 8, "1": 9, "0": 10, "3": 11, "2": 12, "5": 13, "4": 14, "7": 15,
    "6": 16, "9": 17, "8": 18, "=": 19, "A": 20, "@": 21, "C": 22, "B": 23,
    "E": 24, "D": 25, "G": 26, "F": 27, "I": 28, "H": 29, "K": 30, "M": 31,
    "L": 32, "O": 33, "N": 34, "P": 35, "S": 36, "R": 37, "U": 38, "T": 39,
    "W": 40, "V": 41, "Y": 42, "[": 43, "Z": 44, "]": 45, "\\": 46, "a": 47,
    "c": 48, "b": 49, "e": 50, "d": 51, "g": 52, "f": 53, "i": 54, "h": 55,
    "m": 56, "l": 57, "o": 58, "n": 59, "s": 60, "r": 61, "u": 62, "t": 63, "y": 64}

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7,
    "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15,
    "P": 16, "S": 17, "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23,
    "X": 24, "Z": 25}

SMILES_MAXLEN = 100
PROTEIN_MAXLEN = 1000

def label_encode(sequence, charset, max_len):
    encoded = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(sequence[:max_len]):
        encoded[i] = charset.get(ch, 0)
    return encoded

# --- 3. App Helper Functions ---
@st.cache_resource
def load_deepdta_model(dataset):
    model_path = f"models/deepdta_attention_{dataset.lower()}_interpretable.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    
    # We must pass the custom object so Keras knows how to deserialize the layer
    model = tf.keras.models.load_model(model_path, custom_objects={'AttentionPooling1D': AttentionPooling1D})
    return model

def load_sample_data():
    sample_path = "data/sample_pairs.json"
    if os.path.exists(sample_path):
        with open(sample_path, 'r') as f:
            return json.load(f)
    return []

def render_attention_html(sequence, weights, max_len, title):
    # weights shape is (1, max_len, 1), we just need the 1D array up to the actual sequence length
    actual_len = min(len(sequence), max_len)
    w_1d = weights[0, :actual_len, 0]
    
    # Normalize weights so the max weight has 1.0 opacity, making it easier to see
    max_w = np.max(w_1d)
    if max_w > 0:
        w_1d = w_1d / max_w
        
    html = f"<h4>{title}</h4>"
    html += "<div style='font-family: monospace; font-size: 14px; word-wrap: break-word; line-height: 1.8; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f8f9fa; max-height: 400px; overflow-y: auto;'>"
    
    for char, weight in zip(sequence[:actual_len], w_1d):
        # Calculate background color intensity based on weight (using a heatmap color like tomato/red)
        # We use an rgba value where the alpha channel is the normalized weight
        alpha = float(weight)
        html += f"<span style='background-color: rgba(255, 99, 71, {alpha}); padding: 0px 1px; border-radius: 2px;' title='Normalized Weight: {weight:.4f}'>{char}</span>"
    
    html += "</div>"
    return html

# --- 4. Main Streamlit UI ---
st.set_page_config(page_title="DeepDTA + Attention", layout="wide")

# Sidebar
st.sidebar.title("DeepDTA with Attention")
st.sidebar.markdown("""
**Drug-Target Binding Affinity Prediction**

This app reproduces the **DeepDTA** model (Öztürk et al., 2018), predicting how strongly a drug binds to a protein target using only their 1D sequences (no 3D structure needed).

**What's new?**
We added a custom **Attention Pooling** layer. Instead of discarding location information like the original model's max pooling, our model learns which parts of the sequences are most important. 
When it makes a prediction, you can view the attention weights to see exactly which amino acids or SMILES characters it focused on!
""")

# Main Content
st.title("🧪 DeepDTA Binding Affinity Predictor")

# 1. Dataset Selection
dataset = st.selectbox("Select Model/Dataset:", ["Davis", "KIBA"])
model = load_deepdta_model(dataset)

# 2. Sample Data Quick Select
samples = load_sample_data()
selected_sample = None
if samples:
    dataset_samples = [s for s in samples if s['dataset'] == dataset]
    if dataset_samples:
        st.markdown("### Quick Fill Examples")
        cols = st.columns(len(dataset_samples))
        for i, sample in enumerate(dataset_samples):
            with cols[i]:
                if st.button(f"Load {sample['name']}"):
                    st.session_state['smiles'] = sample['smiles']
                    st.session_state['protein'] = sample['protein']

# Initialize session state for inputs if not present
if 'smiles' not in st.session_state:
    st.session_state['smiles'] = ""
if 'protein' not in st.session_state:
    st.session_state['protein'] = ""

# 3. Inputs
st.markdown("### Input Sequences")
smiles_input = st.text_area("Drug SMILES String", value=st.session_state['smiles'], height=100)
protein_input = st.text_area("Protein Sequence (Amino Acids)", value=st.session_state['protein'], height=150)

# 4. Prediction
if st.button("Predict Binding Affinity", type="primary"):
    if not smiles_input or not protein_input:
        st.warning("Please provide both a SMILES string and a Protein sequence.")
    elif model is None:
        st.error("Model failed to load. Cannot run prediction.")
    else:
        with st.spinner("Running inference..."):
            # Encode inputs
            drug_encoded = label_encode(smiles_input, CHARISOSMISET, SMILES_MAXLEN)
            protein_encoded = label_encode(protein_input, CHARPROTSET, PROTEIN_MAXLEN)
            
            # Add batch dimension
            drug_batch = np.expand_dims(drug_encoded, axis=0)
            protein_batch = np.expand_dims(protein_encoded, axis=0)
            
            # Predict
            # The interpretable model outputs: [prediction, drug_attn_weights, protein_attn_weights]
            outputs = model.predict([drug_batch, protein_batch])
            
            prediction = outputs[0][0][0]
            drug_weights = outputs[1]
            protein_weights = outputs[2]
            
            st.success("Prediction complete!")
            
            # Display Prediction
            st.markdown("### Prediction Result")
            if dataset == "Davis":
                st.metric(label="Predicted pKd (higher = stronger affinity)", value=f"{prediction:.3f}")
            else:
                st.metric(label="Predicted KIBA Score (lower = stronger affinity)", value=f"{prediction:.3f}")
            
            st.divider()
            
            # Attention Visualization
            st.markdown("### Attention Weights Interpretation")
            st.markdown("The charts below show which characters/residues the model focused on the most.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(render_attention_html(smiles_input, drug_weights, SMILES_MAXLEN, "Drug (SMILES) Attention"), unsafe_allow_html=True)
            
            with col2:
                st.markdown(render_attention_html(protein_input, protein_weights, PROTEIN_MAXLEN, "Protein Attention"), unsafe_allow_html=True)
