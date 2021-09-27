import shelve
import questionary
import portfolio_builder as pb

def main(display_string=None):
    if display_string is not None:
        print(display_string)

    username = questionary.text("What is your name?").ask()

    with shelve.open('../Resources/shelf') as sh:
        if username in sh:
            message = f"Hello, {username}!"
        else:
            sh[username] = {}
            message = f"It's nice to meet you, {username}!"
            sh.sync()
        print(message)

        if 'portfolio' in sh[username]:
            print('Nice portfolio!', sh[username]['portfolio'])
            if not questionary.confirm("Do you want to keep it?").ask():
                sh[username] = pb.init_portfolio(sh[username])
                sh.sync()

        else:
            print('You need to invest, homie!')
            sh[username] = pb.init_portfolio(sh[username])
            sh.sync()

        print(f"{sh[username]['portfolio']}")
    return username

if __name__ == "__main__":
    username = main('The Four Headless Horsemen')
    print(f"Goodbye, {username}.")