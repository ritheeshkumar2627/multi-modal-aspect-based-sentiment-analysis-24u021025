# Twitter-17 Multimodal Aspect-Based Sentiment Analysis (MABSA)

This project trains a three-class sentiment classifier using both tweet text and its associated image:
- BERT encodes the tweet and target aspect.
- CLIP Vision encodes the image.
- A fusion MLP combines both representations and predicts Negative, Neutral, or Positive.

> Important: the project does not include the Twitter-17 dataset or pretrained model weights. You must obtain the dataset through its authorized source and place the files in the expected folders. Training downloads pretrained BERT/CLIP weights on first run, so an internet connection is normally needed initially.

## 1. Folder layout

Put your dataset files here:

```text
data/raw/
  train.txt
  dev.txt
  test.txt
  images/
    <image files referenced by the annotation files>
```

The loader expects the common Twitter-17 four-line record format:
1. tweet text (often containing `$T$` where the target aspect belongs)
2. target aspect term
3. sentiment label: `-1`, `0`, or `1`
4. image filename

Records are read in groups of four non-empty lines. If your downloaded dataset uses a different format, inspect it and adjust `src/data_loader.py` before training.

## 2. Create an environment

Python 3.10 or 3.11 is recommended.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands from Command Prompt instead:
```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Google Colab

Upload/extract the ZIP into `/content/mabsa_twitter17_project`, upload the dataset into `data/raw/`, then run:
```python
%cd /content/mabsa_twitter17_project
!pip install -r requirements.txt
```
If using Colab, select **Runtime → Change runtime type → T4 GPU** when available.

## 3. Verify the dataset paths

From the project root:
```bash
python check_data.py
```
This reports the number of records, label counts, and whether referenced images exist. Fix any missing paths before training.

## 4. Train

```bash
python run.py
```

By default, training uses a GPU if PyTorch can see one, otherwise CPU. The default batch size is 8 and epochs is 5. The first run downloads `bert-base-uncased` and `openai/clip-vit-base-patch32`.

To change settings, edit `configs/config.py`. On a GPU with limited VRAM, reduce `BATCH_SIZE` to 4 or 2. Training can take a long time because each sample includes an image.

The best checkpoint by development macro-F1 is saved to:
`outputs/best_model.pt`

A log and per-epoch metrics are written to:
`outputs/training_history.csv`

The script evaluates on `dev.txt` each epoch. It does not use `test.txt` for model selection.

## 5. Run the Gradio demo

Train first so `outputs/best_model.pt` exists, then:
```bash
python app_demo.py
```
Open the local URL printed by Gradio (usually `http://127.0.0.1:7860`). Enter the tweet text and target aspect, and optionally upload its image.

For an inference example, you may type:
- Tweet text: `I love the food at $T$`
- Aspect: `food`
- Image: upload an image related to the tweet

If no image is provided, the app uses a zero-valued image tensor. This is a fallback, not equivalent to real image evidence.

## 6. Common issues

- **File not found:** check that files are exactly `data/raw/train.txt`, `data/raw/dev.txt`, `data/raw/test.txt`, and images are in `data/raw/images/`.
- **Missing image warnings:** check the image filename in the fourth line and the actual file extension/name.
- **CUDA out of memory:** lower `BATCH_SIZE` to 4 or 2 in `configs/config.py`.
- **Model download errors:** ensure internet access or configure Hugging Face model caching.
- **No checkpoint:** run `python run.py` successfully before launching the demo.
- **Poor scores:** this is a baseline implementation, not a guaranteed benchmark result. Compare macro-F1 with the benchmark's official evaluation protocol.

## 7. Commands in order

```bash
pip install -r requirements.txt
python check_data.py
python run.py
python app_demo.py
```
