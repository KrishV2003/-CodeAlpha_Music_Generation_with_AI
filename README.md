# Music Generation with LSTM Neural Network

This project demonstrates how to use an LSTM-based(Type of RNN) neural network to generate music using a dataset of classical MIDI files. The output is a MIDI file containing new music composed by the model.

## Features

- Preprocessing MIDI files to extract musical notes and chords.
- Augmenting the dataset by transposing notes to different keys.
- Building an LSTM-based model to learn the patterns in the music.
- Saving and resuming training using checkpoints and last-epoch tracking.
- Generating new music with temperature sampling for creative variations.
- Saving the generated music as a MIDI file.

---

## Installation

### Prerequisites
Ensure the following are installed:
- Python 3.10+
- Libraries: `music21`, `keras`, `tensorflow`, `numpy`

Install the required libraries using pip:
```bash
pip install music21 keras tensorflow numpy
```

---

## Dataset

This project uses a dataset of classical MIDI files. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi).

---

## How to Run

### Step 1: Preprocess MIDI Files
Extract notes and chords from the MIDI files:

```python
notes = load_or_preprocess_notes(dataset_path, notes_file_path)
```
- `dataset_path`: Path to the folder containing MIDI files.
- `notes_file_path`: Path to save the preprocessed notes.

### Step 2: Train the Model
Train the model if fine-tuning or training from scratch is needed:

1. Prepare the sequences for training:
   ```python
   network_input, network_output, n_vocab, int_to_note = prepare_sequences(augmented_notes, sequence_length)
   ```

2. Build the LSTM model:
   ```python
   model = build_model(n_vocab, sequence_length)
   ```

3. Train with checkpoints:
   ```python
   train_with_checkpoint(model, network_input, network_output, epochs=50, batch_size=64)
   ```

### Step 3: Generate New Music

1. Load the pre-trained model:
   ```python
   model = load_model(model_file_path)
   ```

2. Generate music:
   ```python
   prediction_output = generate_advanced_music(model, network_input, int_to_note, n_vocab, sequence_length, num_notes=500)
   ```

3. Save the generated music as a MIDI file:
   ```python
   create_midi(prediction_output, output_midi_path)
   ```

---

## Project Structure

- `preprocess_midi()`: Processes MIDI files to extract notes and chords.
- `transpose_notes()`: Augments the dataset by transposing notes.
- `prepare_sequences()`: Converts notes into sequences for training.
- `build_model()`: Constructs the LSTM-based neural network.
- `train_with_checkpoint()`: Trains the model and saves checkpoints.
- `generate_advanced_music()`: Generates music using the trained model.
- `create_midi()`: Saves the generated music as a MIDI file.

---

## Output

Here is the generated music from this project:

[![Generated Music](https://drive.google.com/file/d/1-8WlxnQmw80HP4amn3Wv9C7qWa6TPEx2/view?usp=drive_link)

---

## Acknowledgments
- [Kaggle Classical Music MIDI Dataset](https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi)
- Libraries: `music21`, `keras`, `tensorflow`
