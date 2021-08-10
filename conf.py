import json

config = None

def load_config():
    print("loading config")
    global config
    try:
        with open("config.json") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        print("Could not file config.json!")
        exit(0)
    except json.decoder.JSONDecodeError as e:
        print("Could not parse config.json")
        print(e)

load_config()
