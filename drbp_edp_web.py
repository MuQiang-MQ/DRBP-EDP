# drbp_edp_web.py

import os
import sys
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import streamlit as st
import altair as alt
import streamlit.components.v1 as components

# ----------------------------
# Logging Configuration
# ----------------------------
log_file_path = os.path.join(os.getcwd(), 'drbp_edp.log')
logging.basicConfig(
    filename=log_file_path,
    level=logging.ERROR,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# ----------------------------
# Device Configuration
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------------
# Application Path
# ----------------------------
def get_application_path():
    if getattr(sys, 'frozen', False):
        # Packaged application
        return os.path.dirname(sys.executable)
    else:
        # Development environment
        return os.path.dirname(os.path.abspath(__file__))


application_path = get_application_path()

# ----------------------------
# Load Tokenizer
# ----------------------------
model_checkpoint = os.path.join(application_path, "model", "esm2_t33_650M_UR50D")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
except Exception as e:
    logging.error("Error loading tokenizer: %s", e)
    st.error(f"Failed to load tokenizer: {e}")
    st.stop()


# ----------------------------
# Define Dataset Class
# ----------------------------
class ProteinSequenceDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=1000):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoded_sequence = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = encoded_sequence['input_ids'].squeeze(0)
        attention_mask = encoded_sequence['attention_mask'].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sequence_length": len(sequence)
        }


# ----------------------------
# Define Model Class
# ----------------------------
class MultiPathProteinClassifier(nn.Module):
    def __init__(self, model_checkpoint):
        super(MultiPathProteinClassifier, self).__init__()
        try:
            self.esm2 = AutoModel.from_pretrained(model_checkpoint, ignore_mismatched_sizes=True)
        except Exception as e:
            logging.error("Error loading ESM2 model: %s", e)
            raise e

        # Path 1: Transformer + CNN
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1280, nhead=8, batch_first=True, dropout=0.6),
            num_layers=1
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1280, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool1d(output_size=500),
            nn.Dropout(0.2),

            nn.Conv1d(in_channels=512, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool1d(output_size=200),
            nn.Dropout(0.2),

            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool1d(output_size=20),
            nn.Dropout(0.2),
        )

        # Path 2: BiLSTM + Attention
        self.bilstm = nn.LSTM(input_size=1280, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)
        self.attention_pool = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.6)
        )

        # Feature Fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(896, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.6),
        )

        # Classifiers for Stage 1 and Stage 2
        self.classifier_stage1 = nn.Linear(256, 1)  # Nucleic acid-binding vs. non-binding
        self.classifier_stage2 = nn.Linear(256, 1)  # DNA-binding vs. RNA-binding

        # Current training stage
        self.current_stage = 1

    def forward(self, input_ids, attention_mask):
        # ESM-2 output
        shared_output = self.esm2(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Path 1: Transformer + CNN
        x1 = self.transformer(shared_output)
        x1 = x1.permute(0, 2, 1)
        x1 = self.cnn(x1).view(x1.size(0), -1)

        # Path 2: BiLSTM + Attention
        x2, _ = self.bilstm(shared_output)
        attention_output, _ = self.attention(x2, x2, x2)
        x2 = attention_output.mean(dim=1)
        x2 = self.attention_pool(x2)

        # Feature Fusion
        features = torch.cat([x1, x2], dim=1)
        features = self.feature_fusion(features)

        # Use the appropriate classifier based on the current stage
        if self.current_stage == 1:
            return self.classifier_stage1(features)
        else:
            return self.classifier_stage2(features)

    def set_stage(self, stage):
        self.current_stage = stage


# ----------------------------
# Load Models with Caching
# ----------------------------
@st.cache_resource
def load_models(model_checkpoint):
    try:
        # Load Stage 1 Model
        stage1_model = MultiPathProteinClassifier(model_checkpoint)
        state_dict_stage1 = torch.load(os.path.join(application_path, 'model', 'best_model_stage1.pth'),
                                       map_location=device)
        # Remove 'module.' prefix if present
        new_state_dict_stage1 = {}
        for k, v in state_dict_stage1.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict_stage1[k] = v
        stage1_model.load_state_dict(new_state_dict_stage1)
        stage1_model.to(device)
        stage1_model.eval()
        stage1_model.set_stage(1)

        # Load Stage 2 Model
        stage2_model = MultiPathProteinClassifier(model_checkpoint)
        state_dict_stage2 = torch.load(os.path.join(application_path, 'model', 'best_model_stage2.pth'),
                                       map_location=device)
        # Remove 'module.' prefix if present
        new_state_dict_stage2 = {}
        for k, v in state_dict_stage2.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict_stage2[k] = v
        stage2_model.load_state_dict(new_state_dict_stage2)
        stage2_model.to(device)
        stage2_model.eval()
        stage2_model.set_stage(2)

        return stage1_model, stage2_model, True
    except Exception as e:
        logging.error("Error loading models: %s", e)
        return None, None, False


# ----------------------------
# Prediction Function with Progress Bar and Percentage
# ----------------------------
def run_prediction(sequences, tokenizer, stage1_model, stage2_model):
    results = []
    dataset = ProteinSequenceDataset(sequences, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        total_sequences = len(sequences)
        progress_bar = st.progress(0)  # Initialize progress bar
        progress_text = st.empty()  # Placeholder for percentage text
        for idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sequence_length = batch['sequence_length'].item()
            sequence = sequences[idx]

            # Stage 1 Prediction: Nucleic acid-binding vs. non-binding
            logits_stage1 = stage1_model(input_ids, attention_mask)
            prob_stage1 = torch.sigmoid(logits_stage1).item()
            pred_stage1 = int(prob_stage1 >= 0.5)

            if pred_stage1 == 1:
                # Stage 2 Prediction: DNA-binding vs. RNA-binding
                logits_stage2 = stage2_model(input_ids, attention_mask)
                prob_stage2 = torch.sigmoid(logits_stage2).item()
                pred_stage2 = int(prob_stage2 >= 0.5)

                if pred_stage2 == 0:
                    prediction = 'DNA-binding protein'
                    prob = 1 - prob_stage2  # Confidence for DNA-binding
                else:
                    prediction = 'RNA-binding protein'
                    prob = prob_stage2  # Confidence for RNA-binding

                prob = max(0.0, min(prob, 1.0))  # Ensure probability is between 0 and 1
            else:
                prediction = 'Non-nucleic acid-binding protein'
                prob = 1 - prob_stage1  # Confidence for non-binding
                prob = max(0.0, min(prob, 1.0))

            # Add sequence length and warnings
            warnings = []
            if sequence_length < 50:
                warnings.append('Length less than 50; prediction may be less reliable.')
            if sequence_length > 1000:
                warnings.append('Length greater than 1000; prediction may be less reliable.')
            length_warning = ' '.join(warnings)

            results.append({
                'Sequence': sequence,
                'Sequence Length': sequence_length,
                'Prediction': prediction,
                'Probability': round(prob, 4),
                'Warning': length_warning
            })

            # Update progress bar and percentage
            progress = (idx + 1) / total_sequences
            progress_bar.progress(min(progress, 1.0))  # Ensure progress does not exceed 100%
            progress_percentage = f"{int(progress * 100)}%"
            progress_text.markdown(f"**Progress:** {progress_percentage}")

    # Clear the progress text after completion
    progress_text.empty()

    return results


# ----------------------------
# Define HTML Component for Copy to Clipboard
# ----------------------------
def create_copy_button(text, button_label="Copy to Clipboard"):
    # Escape backticks and backslashes in the text to prevent breaking the JavaScript
    escaped_text = text.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')

    # Create a unique key for the component
    component_id = f"copy-button-{hash(text)}"

    # JavaScript to copy text to clipboard
    copy_script = f"""
    <script>
    function copyText() {{
        navigator.clipboard.writeText(`{escaped_text}`).then(function() {{
            // Success
            document.getElementById("{component_id}-status").innerText = "Copied!";
        }}, function(err) {{
            // Failure
            document.getElementById("{component_id}-status").innerText = "Failed to copy!";
        }});
    }}
    </script>
    <button onclick="copyText()" style="
        background-color: #7F77CB;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 14px;
    ">{button_label}</button>
    <span id="{component_id}-status" style="margin-left:10px; font-size:14px;"></span>
    """

    components.html(copy_script, height=50)


# ----------------------------
# Callback Functions for Clearing Input and Output
# ----------------------------
def clear_single_input():
    st.session_state['single_sequence'] = ''


def clear_single_output():
    if 'single_result' in st.session_state:
        del st.session_state['single_result']


def clear_batch_input():
    st.session_state['file_upload'] = None
    st.session_state['uploaded_file'] = None
    st.experimental_rerun()  # Reload the app to clear the file uploader


def clear_batch_output():
    if 'batch_result' in st.session_state:
        del st.session_state['batch_result']


# ----------------------------
# Callback Functions for Prediction
# ----------------------------
def predict_single_sequence(stage1_model, stage2_model, tokenizer):
    if not st.session_state['single_sequence'].strip():
        st.error("Please enter a protein sequence.")
    else:
        # Validate Sequence
        valid_aa = 'ACDEFGHIKLMNPQRSTVWY'
        cleaned_seq = ''.join(st.session_state['single_sequence'].upper().split())
        if not cleaned_seq:
            st.error("Sequence is empty after removing spaces and newlines.")
        elif not all(c in valid_aa for c in cleaned_seq):
            invalid_chars = set(c for c in cleaned_seq if c not in valid_aa)
            st.error(
                f"Invalid sequence characters detected: {', '.join(invalid_chars)}\nPlease ensure all sequences only contain valid amino acid characters (ACDEFGHIKLMNPQRSTVWY).")
        else:
            with st.spinner("Predicting, please wait..."):
                results = run_prediction([cleaned_seq], tokenizer, stage1_model, stage2_model)
            if results:
                result = results[0]
                output = (
                    f"**Sequence Length:** {result['Sequence Length']}\n\n"
                    f"**Prediction:** {result['Prediction']}\n\n"
                    f"**Probability:** {result['Probability']:.4f}\n\n"
                    f"**Warning:** {result['Warning']}"
                )
                # Store result in session_state
                st.session_state['single_result'] = output


def predict_batch_sequences(cleaned_sequences, stage1_model, stage2_model, tokenizer):
    with st.spinner("Predicting, please wait..."):
        results = run_prediction(cleaned_sequences, tokenizer, stage1_model, stage2_model)
    if results:
        df_results = pd.DataFrame(results)
        # Store results in session_state
        st.session_state['batch_result'] = df_results


# ----------------------------
# Streamlit Application
# ----------------------------
def main():
    # st.set_page_config(page_title="DRBP-EDP Predictor", layout="wide")

    st.set_page_config(
        page_title="DRBP-EDP Predictor",
        layout="wide",
        page_icon="icon.ico"  # 更换 Favicon
    )

    # Custom CSS for UI Enhancement
    st.markdown("""
    <style>
    .title {
        color: #7F77CB;
        font-size: 70px;
        text-align: center;
        font-weight: bold;
    }
    .section {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 18px;  /* Increase text size */
    }
    .button {
        background-color: #7F77CB;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;  /* Increase button text size */
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="title">DRBP-EDP Predictor</h1>', unsafe_allow_html=True)

    # Add Help Documentation to Sidebar in English
    with st.sidebar:
        st.header("Help Documentation")
        st.markdown("""
        ### How to Use DRBP-EDP Predictor

        1. **Single Sequence Prediction**:
            - Select **"Single Sequence"**.
            - Enter your protein sequence in the text area.
            - Click the **"Predict"** button to initiate prediction.
            - After prediction, click the **"Copy to Clipboard"** button to copy the result.
            - Click the **"Clear Input"** button to clear the input sequence.
            - Click the **"Clear Output"** button to clear the prediction result.


        2. **Batch Prediction via File Upload**:
            - Select **"File Input"**.
            - Upload a file containing protein sequences in **FASTA**, **FA**, **TSV**, or **TXT** format.
                - **FASTA/FA**: Standard FASTA format with description lines starting with **'>'**.
                - **TSV**: Tab-separated values file with a column named **"Sequence"**.
                - **TXT**: Plain text file with one sequence per line.
            - Click the **"Predict"** button to start batch prediction.
            - After prediction, view the prediction statistics and visualization.
            - Click the **"Download Predictions (CSV)"** button to download the results.

        ### Notes
        - Ensure all sequences contain only valid amino acid characters: **ACDEFGHIKLMNPQRSTVWY**.
        - Sequences shorter than **50** or longer than **1000** may have less reliable predictions.
        - The maximum number of sequences allowed per batch is **1000**.
        """)

    # Load Models
    with st.spinner("Loading models, please wait..."):
        stage1_model, stage2_model, success = load_models(model_checkpoint)

    if not success:
        st.error("Failed to load models. Please check the log file for details.")
        st.stop()
    else:
        st.success("Models loaded successfully.")

    # Input Method Selection
    input_method = st.radio("Select Input Method", ("Single Sequence", "File Input"), key='input_method')

    if input_method == "Single Sequence":
        with st.container():
            st.subheader("Enter Protein Sequence")
            # Initialize session_state for 'single_sequence' if not present
            if 'single_sequence' not in st.session_state:
                st.session_state['single_sequence'] = ''
            sequence = st.text_area("Please input your protein sequence here:", height=150, key='single_sequence')

            # Arrange buttons: Predict | Clear Input | Clear Output
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.button("Predict", key='predict_single',
                          on_click=lambda: predict_single_sequence(stage1_model, stage2_model, tokenizer))
            with col2:
                st.button("Clear Input", key='clear_single_input', on_click=clear_single_input)
            with col3:
                st.button("Clear Output", key='clear_single_output', on_click=clear_single_output)

            # Progress bar and percentage below Predict button
            if 'single_result' not in st.session_state and st.session_state['single_sequence'].strip():
                # Only show progress if prediction is ongoing
                pass  # The progress bar is handled within run_prediction

        # Display Prediction Result
        if 'single_result' in st.session_state:
            st.markdown("<div class='section'>" + st.session_state['single_result'] + "</div>", unsafe_allow_html=True)
            # Implement Copy to Clipboard using Custom HTML and JavaScript
            create_copy_button(st.session_state['single_result'], button_label="Copy to Clipboard")

    elif input_method == "File Input":
        with st.container():
            st.subheader("Upload Protein Sequences File")
            # Initialize session_state for 'file_upload' and 'uploaded_file' if not present
            if 'uploaded_file' not in st.session_state:
                st.session_state['uploaded_file'] = None
            uploaded_file = st.file_uploader("Choose a file", type=["fasta", "fa", "tsv", "txt"], key='file_upload')
            if uploaded_file is not None:
                st.session_state['uploaded_file'] = uploaded_file
            else:
                st.session_state['uploaded_file'] = None

            if st.session_state['uploaded_file'] is not None:
                # Parse the uploaded file
                try:
                    if st.session_state['uploaded_file'].name.endswith(('.fasta', '.fa')):
                        sequences = []
                        sequence = ''
                        for line in st.session_state['uploaded_file']:
                            line = line.decode('utf-8').strip()
                            if line.startswith('>'):
                                if sequence:
                                    sequences.append(sequence)
                                    sequence = ''
                            else:
                                sequence += line
                        if sequence:
                            sequences.append(sequence)
                    elif st.session_state['uploaded_file'].name.endswith('.txt'):
                        sequences = [line.decode('utf-8').strip() for line in st.session_state['uploaded_file'] if
                                     line.decode('utf-8').strip()]
                    elif st.session_state['uploaded_file'].name.endswith('.tsv'):
                        df = pd.read_csv(st.session_state['uploaded_file'], sep='\t')
                        if 'Sequence' not in df.columns:
                            st.error("TSV file must contain a 'Sequence' column.")
                            st.stop()
                        sequences = df['Sequence'].dropna().astype(str).tolist()
                    else:
                        st.error("Unsupported file format.")
                        st.stop()
                except Exception as e:
                    logging.error("Error parsing file: %s", e)
                    st.error(f"Failed to parse file: {e}")
                    st.stop()

                if not sequences:
                    st.error("No valid sequences found in the uploaded file.")
                else:
                    # Validate and Clean Sequences
                    cleaned_sequences = []
                    invalid_sequences = []
                    valid_aa = 'ACDEFGHIKLMNPQRSTVWY'
                    for seq in sequences:
                        cleaned_seq = ''.join(seq.upper().split())
                        if not cleaned_seq:
                            continue
                        if all(c in valid_aa for c in cleaned_seq):
                            cleaned_sequences.append(cleaned_seq)
                        else:
                            invalid_sequences.append(seq)

                    if invalid_sequences:
                        displayed_invalid = ', '.join(invalid_sequences[:10])
                        if len(invalid_sequences) > 10:
                            displayed_invalid += '...'
                        st.warning(
                            f"Invalid sequences detected (ignoring invalid characters):\n{displayed_invalid}\n"
                            "Spaces and newlines are ignored. Please ensure all sequences only contain valid amino acid characters (ACDEFGHIKLMNPQRSTVWY)."
                        )
                        if not cleaned_sequences:
                            st.error("No valid sequences to predict after removing invalid characters.")
                            st.stop()

                    # Check Sequence Count Limit
                    max_sequences = 1000
                    if len(cleaned_sequences) > max_sequences:
                        st.error(
                            f"The input file contains {len(cleaned_sequences)} valid sequences, which exceeds the maximum allowed ({max_sequences}). "
                            f"Please select a file with up to {max_sequences} sequences."
                        )
                        st.stop()

                    # Arrange buttons: Predict
                    col1 = st.columns([1])[0]
                    with col1:
                        st.button("Predict", key='predict_batch',
                                  on_click=lambda: predict_batch_sequences(cleaned_sequences, stage1_model,
                                                                           stage2_model, tokenizer))

        # Display Batch Prediction Results
        if 'batch_result' in st.session_state:
            with st.container():
                st.success("Batch prediction successful!")
                df_results = st.session_state['batch_result']

                # Display Prediction Statistics
                total = len(df_results)
                dna = len(df_results[df_results['Prediction'] == 'DNA-binding protein'])
                rna = len(df_results[df_results['Prediction'] == 'RNA-binding protein'])
                non_binding = len(df_results[df_results['Prediction'] == 'Non-nucleic acid-binding protein'])

                st.markdown(f"**Total Sequences:** {total}")
                st.markdown(f"**DNA-binding Proteins:** {dna}")
                st.markdown(f"**RNA-binding Proteins:** {rna}")
                st.markdown(f"**Non-nucleic Acid-binding Proteins:** {non_binding}")

                # Visualization: Prediction Categories Distribution
                prediction_counts = df_results['Prediction'].value_counts().reset_index()
                prediction_counts.columns = ['Prediction', 'Count']

                chart = alt.Chart(prediction_counts).mark_bar().encode(
                    x=alt.X('Prediction:N', title='Prediction Category'),
                    y=alt.Y('Count:Q', title='Number of Sequences', axis=alt.Axis(format='d')),
                    # Ensure integer display
                    color='Prediction:N'
                ).properties(
                    width=600,
                    height=400,
                    title='Prediction Categories Distribution'
                )

                st.altair_chart(chart)

                # Provide Download Button for CSV
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions (CSV)",
                    data=csv,
                    file_name="predictions.csv",
                    mime='text/csv',
                )

    # ----------------------------
    # Display Log File (Optional)
    # ----------------------------
    with st.expander("Show Log File"):
        try:
            with open('drbp_edp.log', 'r') as log_file:
                log_contents = log_file.read()
                st.text_area("Log File", log_contents, height=300)
        except FileNotFoundError:
            st.write("Log file not found.")


if __name__ == "__main__":
    main()
