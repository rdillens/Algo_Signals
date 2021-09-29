import questionary
import shelve


def get_username():
    username = questionary.text("What is your name?").ask()
    with shelve.open('../Resources/shelf') as sh:
        # Check to see if username exists in shelf
        if username in sh:
            message = f"Hello, {username}!"
        # If username does not exist, create empty dictionary
        else:
            sh[username] = {}
            message = f"It's nice to meet you, {username}!"
            sh.sync()
        print(message)

    return username




