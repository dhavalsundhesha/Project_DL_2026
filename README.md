# Group 3 — Handwritten Digit Recognition using Four Algorithms

## Algorithms implemented
1. MLP (fully connected neural network)
2. LeNet-5-style CNN
3. Custom CNN
4. Small ResNet-style CNN

All four are implemented in TensorFlow/Keras and trained/evaluated on MNIST. The Streamlit app lets you select a trained model, upload or draw a digit, view the predicted class and probability chart, and view Grad-CAM when the architecture supports it.

## Setup (Mac)
Run these commands from this folder. Use the same Python environment that has TensorFlow installed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py
```

If TensorFlow is already installed in `~/.venv`, activate that environment instead:
```bash
source ~/.venv/bin/activate
cd ~/Downloads/Group_3_Handwritten_Digit_4_Algorithms
python train_models.py
python -m streamlit run app.py
```

## Outputs
`models/` contains the trained `.keras` files. `results/` contains:
- model_comparison.csv and model_comparison.json
- per-model confusion matrix CSV
- per-model classification report
- per-model training/validation accuracy and loss plots
- per-model training history

## Important for report
Do not write metric values until `train_models.py` has finished. Copy actual values from `results/model_comparison.csv`. The script uses the MNIST official 60,000-image training set with a 10% validation split and the official 10,000-image test set. Default epochs: 5; batch size: 128; Adam learning rate: 0.001; loss: sparse categorical cross-entropy.
