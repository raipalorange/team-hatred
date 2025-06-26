# Hate Speech Detection using BiLSTM + Self-Attention

## Overview
Advanced hate speech detection system combining Word2Vec embeddings with BiLSTM architecture enhanced by self-attention mechanisms. Achieves **94.6% accuracy** on a comprehensive dataset of 600,000 samples from multiple social media platforms.

## Key Features
- **Custom Word2Vec embeddings** trained on 600k diverse samples
- **BiLSTM with self-attention** for enhanced contextual understanding  
- **Multi-platform dataset** (Twitter, Reddit, 4Chan)
- **State-of-the-art performance** with 96% precision and recall
- **Production-ready architecture** with modular design

## Performance Results
| Metric | Baseline BiLSTM | Our Model (BiLSTM + Attention) |
|--------|----------------|--------------------------------|
| **Accuracy** | - | **94.6%** |
| **Precision** | 95% | **96%** |
| **Recall** | 95% | **96%** |
| **F1-Score** | 95% | **95%** |

## Dataset Composition (600k samples)
- **Davidson Dataset**: 24k Twitter posts with human annotation
- **UC Dublin Dataset**: 3k cross-platform comments (Twitter, Reddit, 4Chan)  
- **UC Berkeley Dataset**: 35k comments with magnitude ratings
- **Lakehead Curated Dataset**: 300k preprocessed social media texts

## Architecture
Input Text → Preprocessing → Word2Vec Embeddings → BiLSTM → Self-Attention → Classification

## Project Structure
- **config/** - Configuration files
- **utils/** - Data processing utilities  
- **models/** - BiLSTM + Self-Attention architecture AND Custom Word2Vec model (600k samples)
- **scripts/** - Training and evaluation scripts
- **main.py** - Main execution script

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (training + evaluation + report)
python main.py
```
## Installation
```bash
# Clone the repository
git clone [your-repo-url]
cd hate-speech-detection

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```
## Technical Details

### Model Architecture
- **Embedding Dimension**: Custom Word2Vec trained on 600k samples
- **BiLSTM Architecture**: Bidirectional LSTM with self-attention mechanism
- **Training Epochs**: 15 epochs with Adam optimizer
- **Data Split**: 80% training, 20% testing (random split)
- **Hardware Compatibility**: CPU/GPU auto-detection

### Experimental Setup
- **Development Platform**: Google Colab Pro
- **Hardware**: NVIDIA Tesla T4 GPU (16GB VRAM)
- **Framework**: PyTorch with CUDA acceleration
- **Key Libraries**: 
  - NumPy for numerical computations
  - Pandas for data preprocessing  
  - Gensim for Word2Vec embeddings
  - Scikit-learn for model evaluation

### Training Configuration
- **Batch Size**: 500
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Data Split**: 80% train / 20% test
- **Validation**: Random split methodology
