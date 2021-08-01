#!/usr/bin/python3

##### based on https://www.tensorflow.org/text/tutorials/text_generation

import datetime
from os import listdir, path
import time
import sys
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

TRUNCATE_TEXT = False
# TRUNCATE_TEXT = 10000

checkpoint_dir = 'training_checkpoints'


class ChatGenerator:
    def __init__(self, model, vocab, chars_from_ids, ids_from_chars, loaded_checkpoint):
        self.model = model
        self.vocab = vocab
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars
        self.loaded_checkpoint = loaded_checkpoint
        self.temperature = 1.0

        self.one_step_model = None

    def generate_reply(self, message):
        if not self.one_step_model:
            model = self.model
            chars_from_ids = self.chars_from_ids
            ids_from_chars = self.ids_from_chars
            self.one_step_model = OneStep(model, chars_from_ids, ids_from_chars, temperature=self.temperature)

        states = None
        next_char = tf.constant([bytes(message, 'utf-8')])
        result = [next_char]

        for n in range(500):
            next_char, states = self.one_step_model.generate_one_step(next_char, states=states)
            if next_char == '\n':
                break
            result.append(next_char)

        result = tf.strings.join(result)
        result = result[0].numpy().decode('utf-8')
        print("result from model: " + result)

        return result.split("\n")[-1]


    def set_temperature(self, temperature):
        self.temperature = temperature
        self.initialize()


    def initialize(self):
        self.one_step_model = OneStep(
            model=self.model,
            chars_from_ids=self.chars_from_ids,
            ids_from_chars=self.ids_from_chars,
            temperature=self.temperature
        )



class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(
        rnn_units,
        return_sequences=True,
        return_state=True
    )
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x


def run_id():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def initialize_generator(load_checkpoint=None):
    # TODO limit the vocab to just string.printable
    # also TODO try using words instead of chars as the vocab
    vocab_filename = "vocab.txt"
    vocab = open(vocab_filename, 'rb').read().decode('utf-8')
    vocab = list(sorted(set(vocab)))
    vocab_size = len(vocab)

    assert vocab_size == 654

    ids_from_chars = preprocessing.StringLookup(vocabulary=vocab, mask_token=None)
    chars_from_ids = preprocessing.StringLookup(vocabulary=vocab, invert=True, mask_token=None)

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024

    model = MyModel(
        # Be sure the vocabulary size matches the `StringLookup` layers.
        vocab_size=vocab_size + 1,  # add one for [UNK]??
        embedding_dim=embedding_dim,
        rnn_units=rnn_units
    )

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)

    if load_checkpoint:
        if load_checkpoint == "latest":
            print("*** loading latest checkpoint")
            latest_checkpoint = None
            checkpoint_files = [f for f in listdir(checkpoint_dir) if path.isfile(path.join(checkpoint_dir, f))]
            for checkpoint_file in checkpoint_files:
                if not checkpoint_file.endswith(".index"):
                    continue
                checkpoint_file = path.join(checkpoint_dir, checkpoint_file)
                checkpoint_file = checkpoint_file[0 : -1 * len(".index")]
                if latest_checkpoint is None or checkpoint_file > latest_checkpoint:
                    latest_checkpoint = checkpoint_file
            load_checkpoint = latest_checkpoint

        print("*** loading checkpoint file " + load_checkpoint)
        model.load_weights(load_checkpoint)

    return ChatGenerator(
        model=model,
        vocab=vocab,
        chars_from_ids=chars_from_ids,
        ids_from_chars=ids_from_chars,
        loaded_checkpoint=load_checkpoint,
    )


def generate_model(logs_folder, load_checkpoint=None, epochs=1):
    chat_generator = initialize_generator(load_checkpoint)
    model = chat_generator.model
    vocab = chat_generator.vocab
    ids_from_chars = chat_generator.ids_from_chars
    chars_from_ids = chat_generator.chars_from_ids

    in_folder = logs_folder
    text = ""
    input_files = [f for f in listdir(in_folder) if path.isfile(path.join(in_folder, f))]
    for filename in input_files:
        text += open(path.join(in_folder, filename), 'rb').read().decode('utf-8')

    if TRUNCATE_TEXT:
        truncate_start = random.randrange(0, len(text) - TRUNCATE_TEXT)
        text = text[truncate_start:truncate_start + TRUNCATE_TEXT]

    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    seq_length = 100
    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    BATCH_SIZE = 64

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    # Directory where the checkpoints will be saved
    # Name of the checkpoint files
    checkpoint_prefix = path.join(checkpoint_dir, "ckpt_" + run_id() + "_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True,
        verbose=1,
    )

    history = model.fit(
        dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback]
    )

    print(chat_generator.generate_reply("hi\n"))


class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(
        inputs=input_ids,
        states=states,
        return_state=True
    )
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits / self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states

if __name__ == "__main__":
    generate_model(
        logs_folder=sys.argv[1],
        load_checkpoint=sys.argv[2] if len(sys.argv) >= 3 else None,
        epochs=int(sys.argv[3]) if len(sys.argv) >= 4 else 1,
    )
