#!/usr/bin/python3

from os import listdir
from os.path import isfile, join
import sys
import string

replace_strings = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
}

def process_one(in_filename, out_filename):
    infile = open(in_filename, encoding='utf-8')
    outfile = open(out_filename, 'w', encoding='utf-8')

    while True:
        try:
            line = infile.readline()
            if not line:
                break
        except UnicodeDecodeError:
            continue

        line = line.strip()

        parts = line.split(' ')
        timestamp = parts[0]
        username = parts[1]

        if username[0] != '<' or username[-1] != '>':
            continue

        if len(parts) <= 2:
            continue

        # if username == '<gonzobot>' or (len(parts[2]) > 0 and parts[2][0]) == '.':
        #     continue

        message_parts = parts[2:]  # strip the username
        message = " ".join(message_parts)

        for key, val in replace_strings.items():
            message = message.replace(key, val)

        if any([c not in string.printable for c in message]):
            # skip lines with emojis or other unusual characters
            continue

        outfile.write(message + '\n')


in_folder = sys.argv[1]
out_folder = sys.argv[2]

input_files = [f for f in listdir(in_folder) if isfile(join(in_folder, f))]
for filename in input_files:
    in_filename = join(in_folder, filename)
    out_filename = join(out_folder, filename)
    print(in_filename, out_filename)
    process_one(in_filename, out_filename)
