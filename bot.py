#!/usr/bin/python3

import re
import sys
import time
import traceback

from irc import *
from generate_model import initialize_generator
from conf import config, load_config

irc = IRC()
irc.connect()

chat_generator = initialize_generator("latest")
chat_generator.initialize()

botnick = config['irc']['nick']

count_since_response = 100
last_respone_time = int(time.time())

def message_handler(username, channel, message, full_user):
    global count_since_response
    if not should_respond(message):
        count_since_response += 1
        return

    input_str = ""
    while len(input_str) < 100:
        input_str += message + "\n"

    reply = chat_generator.generate_reply(input_str)
    if botnick.upper() in message.upper():
        reply = username + ": " + reply
    irc.send_to_channel(channel, reply)

    count_since_response = 0
    last_respone_time = int(time.time())


def should_respond(message):
    if botnick.upper() in message.upper():
        return True

    if 'respond_without_prompt' in config:
        global count_since_response
        if count_since_response < config['respond_without_prompt']['messages_between']:
            # too few messages
            return False

        time_since_last_response = int(time.time()) - last_respone_time
        if time_since_last_response < config['respond_without_prompt']['seconds_since_last_response']:
            # last response too recent
            return False
        return True

    return False

def admin_commands(username, channel, message, full_user):
    if full_user != config['admin']:
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

    elif command in ["reload_model", "reloadmodel", "model"]:
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

    elif command in ["temp", "temperature"]:
        if not args:
            irc.send_to_channel(channel, username + ": current temperature is " + str(chat_generator.temperature))
            return

        try:
            temperature = float(args)
            assert temperature > 0
            old_temperature = chat_generator.temperature
            chat_generator.set_temperature(temperature)
            irc.send_to_channel(channel, username + ": set temperature to " + str(temperature) + " (was " + str(old_temperature) + ")")
        except Exception as e:
            print(e)
            irc.send_to_channel(channel, username + ": invalid temperature")

    elif command in ["reload_config", "config", "reloadconfig"]:
        load_config()
        irc.send_to_channel(channel, username + ": reloaded config.json. new command_key = " + config['command_key'])


irc.add_message_handler(message_handler)
irc.add_message_handler(admin_commands)
