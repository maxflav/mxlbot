#!/usr/bin/python3

import re
import sys
import time
import traceback

from irc import *
from generate_model import initialize_generator
from conf import config

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

irc = IRC()
irc.connect()

chat_generator = initialize_generator("latest")
chat_generator.initialize()


def message_handler(username, channel, message, full_user):
    if message.upper().startswith(botnick.upper() + " "):
        message = message[len(botnick + " "):]
    elif message.upper().startswith(botnick.upper() + ": "):
        message = message[len(botnick + ": "):]
    elif botnick.upper() not in message.upper():
        return

    reply = chat_generator.generate_reply(message + "\n")
    reply = maybe_strip_username(reply)
    irc.send_to_channel(channel, username + ": " + reply)


def maybe_strip_username(reply):
    # TODO: strip usernames from training logs and remove this

    reply_parts = reply.split(" ")
    if len(reply_parts) == 1:
        return reply

    username = reply_parts[0]
    if not username.startswith("<") and not username.endswith(">"):
        return reply

    if len(username) > 30:
        return reply

    reply_parts = reply_parts[1:]
    return " ".join(reply_parts)



def admin_commands(username, channel, message, full_user):
    if full_user != admin:
        return

    if not message.startswith(config['command_key']):
        return

    parts = message.split(" ")
    command = parts[0][len(config['command_key']):]
    args = "".join(parts[1:])

    if command == "join":
        irc.send("JOIN " + args + "\n")

    elif command in ["leave", "part"]:
        to_leave = args if args else channel
        irc.send("PART " + to_leave + "\n")

    elif command == "reload":
        global chat_generator
        to_load = args if args else "latest"
        try:
            chat_generator = initialize_generator(to_load)
            chat_generator.initialize()
            irc.send_to_channel(channel, username + ": reloaded model " + chat_generator.loaded_checkpoint)
        except:
            traceback.print_exc()
            irc.send_to_channel(channel, username + ": couldn't load model " + args)

    elif command == "shutdown":
        irc.stop()

    elif command == "temp":
        if not args:
            irc.send_to_channel(channel, username + ": current temperature is " + str(old_temperature))

        try:
            temperature = float(args)
            assert temperature > 0
            old_temperature = chat_generator.temperature
            chat_generator.set_temperature(temperature)
            irc.send_to_channel(channel, username + ": set temperature to " + str(temperature) + " (was " + str(old_temperature) + ")")
        except Exception as e:
            print(e)
            irc.send_to_channel(channel, username + ": invalid temperature")


irc.add_message_handler(message_handler)
irc.add_message_handler(admin_commands)
