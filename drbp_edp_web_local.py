# drbp_edp_web_local.py

import os
import sys
import logging
import traceback

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import streamlit as st
import altair as alt
import streamlit.components.v1 as components
from datetime import datetime

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


def get_cleaned_sequences():
    """Process uploaded file and return cleaned sequence list"""
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')

    try:
        # Read content from uploaded file
        content = st.session_state.uploaded_file.getvalue().decode('utf-8')
    except UnicodeDecodeError:
        st.error("File decoding error (only UTF-8 text files supported)")
        return []
    except Exception as e:
        logging.error(f"File reading failed: {str(e)}")
        st.error("File processing error. Please check logs")
        return []

    sequences = []
    current_seq = []
    line_count = 0

    # Parse FASTA format
    for line in content.splitlines():
        line = line.strip()
        line_count += 1

        # Process header lines
        if line.startswith('>'):
            if current_seq:
                full_seq = ''.join(current_seq).upper()
                cleaned_seq = ''.join([c for c in full_seq if c in valid_aa])

                # Validity check
                if len(cleaned_seq) < 1:
                    st.warning(f"Line {line_count} became empty after cleaning, skipped")
                    continue

                sequences.append(cleaned_seq)
                current_seq = []
            continue

        # Process sequence lines
        cleaned_line = line.replace(" ", "").replace("\n", "")
        current_seq.append(cleaned_line)

    # Process final sequence
    if current_seq:
        full_seq = ''.join(current_seq).upper()
        cleaned_seq = ''.join([c for c in full_seq if c in valid_aa])
        if len(cleaned_seq) >= 1:
            sequences.append(cleaned_seq)

    # Post-processing check
    if len(sequences) == 0:
        st.error("No valid sequences found in file")
        return []

    # Display cleaning statistics
    st.info(f"Found {len(sequences)} valid sequences (original lines: {line_count})")

    # Deduplication
    unique_seqs = list({seq: None for seq in sequences}.keys())
    if len(unique_seqs) < len(sequences):
        st.warning(f"Removed {len(sequences) - len(unique_seqs)} duplicate sequences")

    return unique_seqs

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

@st.cache_resource
def load_models(model_checkpoint):
    try:
        # Construct file paths
        stage1_model_path = os.path.join(application_path, 'model', 'best_model_stage1.pth')
        stage2_model_path = os.path.join(application_path, 'model', 'best_model_stage2.pth')

        # Debugging: Check if model files exist
        if not os.path.exists(stage1_model_path):
            logging.error(f"Stage 1 model file not found: {stage1_model_path}")
            return None, None, False
        if not os.path.exists(stage2_model_path):
            logging.error(f"Stage 2 model file not found: {stage2_model_path}")
            return None, None, False

        # Load Stage 1 Model
        stage1_model = MultiPathProteinClassifier(model_checkpoint)
        state_dict_stage1 = torch.load(stage1_model_path, map_location=device)
        # Remove 'module.' prefix if present
        new_state_dict_stage1 = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict_stage1.items()}
        stage1_model.load_state_dict(new_state_dict_stage1)
        stage1_model.to(device)
        stage1_model.eval()
        stage1_model.set_stage(1)

        # Load Stage 2 Model
        stage2_model = MultiPathProteinClassifier(model_checkpoint)
        state_dict_stage2 = torch.load(stage2_model_path, map_location=device)
        # Remove 'module.' prefix if present
        new_state_dict_stage2 = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict_stage2.items()}
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

def run_prediction(sequences, tokenizer, model, mode):
    """通用预测函数"""
    results = []
    dataset = ProteinSequenceDataset(sequences, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        total = len(sequences)
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seq_len = batch['sequence_length'].item()
            seq = sequences[idx]

            # 统一预测逻辑
            logits = model(input_ids, attention_mask)
            prob = torch.sigmoid(logits).item()
            pred = int(prob >= 0.5)

            # 根据模式生成预测结果
            if mode == "nucleic":
                prediction = 'Nucleic acid-binding protein' if pred else 'Non-nucleic acid-binding protein'
            else:
                prediction = 'RNA-binding protein' if pred else 'DNA-binding protein'

            confidence = prob if pred else 1 - prob
            confidence = max(0.0, min(confidence, 1.0))

            # 警告生成（保持原有逻辑）
            warnings = []
            if seq_len < 50:
                warnings.append('Length less than 50; prediction may be less reliable.')
            if seq_len > 1000:
                warnings.append('Length greater than 1000; prediction may be less reliable.')

            # 可靠性评估
            reliability = "High" if confidence > 0.7 else "Medium" if confidence > 0.5 else "Low"

            results.append({
                'Sequence': seq,
                'Length': seq_len,
                'Prediction': prediction,
                'Probability': round(confidence, 4),
                'Reliability': reliability,
                'Warning': '; '.join(warnings) if warnings else 'None'
            })

            # 更新进度
            progress = (idx + 1) / total
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {int(progress * 100)}%")

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
    """处理单序列预测"""
    if not st.session_state['single_sequence'].strip():
        st.error("Please enter a protein sequence.")
        return

    # 获取当前预测模式
    current_mode = "nucleic" if "nabp" in st.session_state.pred_mode.lower() else "dna_rna"
    model = stage1_model if current_mode == "nucleic" else stage2_model

    # 异常处理：模型加载检查
    if model is None:
        st.error(f"{current_mode} model not loaded properly")
        return

    # 验证序列有效性
    valid_aa = 'ACDEFGHIKLMNPQRSTVWY'
    cleaned_seq = ''.join(st.session_state['single_sequence'].upper().split())

    if not cleaned_seq:
        st.error("Sequence is empty after removing spaces and newlines.")
        return

    invalid_chars = set(c for c in cleaned_seq if c not in valid_aa)
    if invalid_chars:
        st.error(f"Invalid sequence characters detected: {', '.join(invalid_chars)}\nPlease ensure all sequences only contain valid amino acid characters (ACDEFGHIKLMNPQRSTVWY).")
        return

    with st.spinner("Predicting..."):
        try:
            results = run_prediction([cleaned_seq], tokenizer, model, current_mode)
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            st.error("Prediction failed, check log for details")
            return

    if results:
        result = results[0]
        output = (
            f"**Sequence Length:** {result['Length']}\n\n"
            f"**Prediction:** {result['Prediction']}\n\n"
            f"**Probability:** {result['Probability']:.4f}\n\n"
            f"**Reliability:** {result['Reliability']}\n\n"
            f"**Warning:** {result['Warning']}"
        )
        st.session_state['single_result'] = output


def predict_batch_sequences(cleaned_sequences, stage1_model, stage2_model, tokenizer):
    if not isinstance(cleaned_sequences, list) or len(cleaned_sequences) == 0:
        st.error("Invalid input sequences")
        return

    current_mode = "nucleic" if "nabp" in st.session_state.pred_mode.lower() else "dna_rna"

    try:
        model = stage1_model if current_mode == "nucleic" else stage2_model
    except AttributeError:
        st.error("Model loading failed")
        return


    if len(cleaned_sequences) > 1000:
        st.error("Exceeds maximum 1000 sequences")
        return

    # 预测执行（保持原逻辑）
    with st.spinner(f"Processing {len(cleaned_sequences)} sequences..."):
        try:
            # 实际调用的是 forward 方法，不需要 predict 方法
            results = run_prediction(cleaned_sequences, tokenizer, model, current_mode)
        except Exception as e:
            logging.error(f"Prediction Error: {traceback.format_exc()}")
            st.error("Prediction failed - check logs")
            return

    # 结果存储（保持不变）
    if results:
        st.session_state.batch_result = pd.DataFrame(results).assign(
            Mode=current_mode,
            Timestamp=datetime.now().isoformat()
        )
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
            - Select the classification task.
            - Enter your protein sequence in the text area.
            - Click the **"Predict"** button to initiate prediction.
            - After prediction, click the **"Copy to Clipboard"** button to copy the result.
            - Click the **"Clear Input"** button to clear the input sequence.
            - Click the **"Clear Output"** button to clear the prediction result.


        2. **Batch Prediction via File Upload**:
            - Select **"File Input"**.
            - Select the classification task.
            - Upload a file containing protein sequences in **FASTA**, **FA**, **TSV**, or **TXT** format.
                - **FASTA/FA**: Standard FASTA format with description lines starting with **'>'**.
                - **TSV**: Tab-separated values file with a column named **"Sequence"**.
                - **TXT**: Plain text file with one sequence per line.
            - Click the **"Predict"** button to start batch prediction.
            - After prediction, view the prediction statistics and visualization.
            - Click the **"Download Predictions (CSV)"** button to download the results.

        ### Notes
        - **The classification task:**
            - **Nucleic acid-binding proteins (NABPs) / Non-nucleic acid-binding proteins (Non-NABPs)**: Binary classification distinguishing NABPs from non-NABPs.
            - **DNA-binding proteins (DBPs) / RNA-binding proteins (RBPs)**: Fine-grained classification differentiating DBPs and RBPs within NABPs.
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
    st.markdown("### Select Input Method")
    input_method = st.radio("Choose method type:", ("Single Sequence", "File Input"), key='input_method')
    st.markdown("### Select Classification Task")
    prediction_mode = st.radio(
        "Choose prediction type:",
        ("NABPs  /  Non-NABPs",
         "DBPs  /  RBPs"),
        index=0,
        key='pred_mode'
    )

    # 根据模式显示对应说明
    if "nabp" in prediction_mode.lower():
        st.info("""
            **1**: Nucleic Acid-Binding Proteins Prediction
            - Predicts whether a protein is nucleic acid-binding (DNA/RNA) or not
            - Output categories: Nucleic acid-binding protein / Non-nucleic acid-binding protein
            """)
    else:
        st.info("""
            **2**: DNA/RNA-Binding Specificity Prediction
            - Predicts the nucleic acid binding specificity of nucleic acid-binding proteins
            - Output categories: DNA-binding protein / RNA-binding protein
            """)

    # 根据选择加载对应模型
    if "nucleic" in prediction_mode.lower():
        current_model = stage1_model
        model_type = "nucleic"
    else:
        current_model = stage2_model
        model_type = "dna_rna"

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
            uploaded_file = st.file_uploader("Choose a file", type=["fasta", "fa", "tsv", "txt"], key='file_upload', on_change=lambda: st.session_state.pop('batch_result', None))

            # 更新session_state
            st.session_state['uploaded_file'] = uploaded_file

            if st.session_state['uploaded_file'] is not None:
                # File parsing logic
                try:
                    file_ext = st.session_state['uploaded_file'].name.lower()
                    sequences = []

                    if file_ext.endswith(('.fasta', '.fa')):
                        current_seq = []
                        for line in st.session_state['uploaded_file']:
                            line = line.decode('utf-8').strip()
                            if line.startswith('>'):
                                if current_seq:
                                    sequences.append(''.join(current_seq))
                                    current_seq = []
                            else:
                                current_seq.append(line)
                        if current_seq:
                            sequences.append(''.join(current_seq))

                    elif file_ext.endswith('.txt'):
                        sequences = [
                            line.decode('utf-8').strip()
                            for line in st.session_state['uploaded_file']
                            if line.decode('utf-8').strip()
                        ]

                    elif file_ext.endswith('.tsv'):
                        df = pd.read_csv(st.session_state['uploaded_file'], sep='\t')
                        if 'Sequence' not in df.columns:
                            st.error("TSV file requires 'Sequence' column")
                            st.stop()
                        sequences = df['Sequence'].dropna().astype(str).tolist()

                    else:
                        st.error("Unsupported file format")
                        st.stop()

                except Exception as e:
                    logging.error("File parsing error: %s", e)
                    st.error(f"File processing failed: {str(e)}")
                    st.stop()

                if not sequences:
                    st.error("No valid sequences found in file")
                else:
                    # Sequence validation
                    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
                    cleaned_sequences = []
                    invalid_entries = []

                    for seq in sequences:
                        cleaned_seq = seq.upper().replace(' ', '')
                        if not cleaned_seq:
                            continue
                        if all(c in valid_aa for c in cleaned_seq):
                            cleaned_sequences.append(cleaned_seq)
                        else:
                            invalid_entries.append(seq)

                    # Handle invalid sequences
                    if invalid_entries:
                        display_limit = 10
                        sample_invalid = ', '.join(invalid_entries[:display_limit])
                        if len(invalid_entries) > display_limit:
                            sample_invalid += '...'

                        st.warning(
                            f"Validation failed for {len(invalid_entries)} sequences (spaces/newlines removed):\n"
                            f"Invalid samples: {sample_invalid}\n"
                            "Required format: Sequences must exclusively contain uppercase "
                            "amino acid codes (ACDEFGHIKLMNPQRSTVWY)"
                        )
                        if not cleaned_sequences:
                            st.error("No valid sequences to predict after removing invalid characters.")
                            st.stop()

                    # Sequence quantity check
                    max_sequences = 1000
                    if len(cleaned_sequences) > max_sequences:
                        st.error(
                            f"Exceeded maximum sequence limit ({max_sequences})\n"
                            f"Found: {len(cleaned_sequences)} valid sequences"
                        )
                        st.stop()

                    # Prediction trigger
                    st.button(
                        "Run Prediction",
                        on_click=lambda: predict_batch_sequences(
                            cleaned_sequences,  # 显式传入清洗后的序列
                            stage1_model,
                            stage2_model,
                            tokenizer
                        ),
                        key="predict_batch_v2"
                    )

        # Display Batch Prediction Results
        if 'batch_result' in st.session_state:
            with st.container():
                st.success("Batch prediction completed!")
                df_results = st.session_state['batch_result']

                # Dynamic label configuration
                current_mode = st.session_state.get('current_mode', 'nucleic')

                if current_mode == 'nucleic':
                    positive_label = "Nucleic acid-binding protein"
                    negative_label = "Non-nucleic acid-binding protein"
                    category_title = "Binding Category"
                else:
                    positive_label = "DNA-binding protein"
                    negative_label = "RNA-binding protein"
                    category_title = "Binding Type"

                # Statistical analysis
                total = len(df_results)
                positive_count = len(df_results[df_results['Prediction'] == positive_label])

                # Handle DNA/RNA mode differently
                if current_mode == 'dna_rna':
                    negative_count = len(df_results[df_results['Prediction'] == negative_label])
                    non_binding = total - positive_count - negative_count
                else:
                    non_binding = len(df_results[df_results['Prediction'] == negative_label])

                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Sequences", total)
                with col2:
                    st.metric(positive_label, positive_count)
                with col3:
                    if current_mode == 'dna_rna':
                        st.metric(negative_label, negative_count)
                    else:
                        st.metric(negative_label, non_binding)

                # Dynamic visualization
                prediction_counts = df_results['Prediction'].value_counts().reset_index()
                prediction_counts.columns = ['Category', 'Count']

                chart = alt.Chart(prediction_counts).mark_bar().encode(
                    x=alt.X('Category:N',
                            title=category_title,
                            sort=alt.EncodingSortField(field='Count', order='descending')),
                    y=alt.Y('Count:Q',
                            title='Number of Sequences',
                            axis=alt.Axis(format='d')),
                    color=alt.Color('Category:N',
                                    legend=alt.Legend(title="Prediction Categories"),
                                    scale=alt.Scale(scheme='category10'))
                ).properties(
                    width=700,
                    height=450,
                    title=f'Distribution of Prediction Results ({current_mode.upper()} Mode)'
                ).configure_axis(
                    labelFontSize=12,
                    titleFontSize=14
                ).configure_title(
                    fontSize=16,
                    anchor='start'
                )

                st.altair_chart(chart)

                # Enhanced data export
                csv = df_results.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="Download Full Results (CSV)",
                    data=csv,
                    file_name=f"predictions_{current_mode}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime='text/csv',
                    help="Download contains all prediction results with timestamps"
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
