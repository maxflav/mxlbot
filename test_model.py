#!/usr/bin/python3

import argparse
from os import listdir, path
import sys

from generate_model import initialize_generator, DEFAULT_RNN_UNITS, DEFAULT_SEQ_LENGTH, DEFAULT_EMBEDDING_DIM

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input phrase. default="hi"', default='hi')
parser.add_argument('-n', type=int, help='number of tests to run. default=1', default=1)
parser.add_argument(
    '--model',
    help='model to test. "latest" to use the latest or "all" to test all models in checkpoint-dir. default="latest"',
    default="latest",
)
parser.add_argument(
    '--dir',
    help='checkpoint dir to use. default="training_checkpoints"',
    default="training_checkpoints",
)
parser.add_argument(
    '--vocab',
    help='vocab file to use. default=None, which uses string.printable',
    default=None,
)
parser.add_argument('--embedding_dim', default=DEFAULT_EMBEDDING_DIM)
parser.add_argument('--rnn_units', default=DEFAULT_RNN_UNITS)
parser.add_argument('--seq_length', default=DEFAULT_SEQ_LENGTH)


args = parser.parse_args()

models = [args.model]
if args.model == "all":
    models = []
    checkpoint_files = [f for f in listdir(args.dir) if path.isfile(path.join(args.dir, f))]
    checkpoint_files = sorted(checkpoint_files)
    for checkpoint_file in checkpoint_files:
        if not checkpoint_file.endswith(".index"):
            continue
        checkpoint_file = path.join(args.dir, checkpoint_file)
        checkpoint_file = checkpoint_file[0 : -1 * len(".index")]
        models.append(checkpoint_file)

input_str = ""
while len(input_str) < 100:
    input_str += args.input + "\n"

for model_to_test in models:
    print("\n\ntrying model: " + model_to_test + "\n")
    try:
        chat_generator = initialize_generator(
            model_to_test,
            args.vocab,
            embedding_dim=int(args.embedding_dim),
            rnn_units=int(args.rnn_units),
            seq_length=int(args.seq_length),
        )
        chat_generator.initialize()

        for n in range(args.n):
            print(chat_generator.generate_reply(input_str))
    except Exception as e:
        print("** Model failed **")
        import traceback
        traceback.print_exc()
