import shelve
import questionary
import portfolio_builder as pb

def main(display_string=None):
    if display_string is not None:
        print(display_string)

    sh = shelve.open('../Resources/shelf')

    username = questionary.text("What is your name?").ask()

    if username in sh:
        message = f"Hello, {username}!"
    else:
        sh[username] = {}
        message = f"It's nice to meet you, {username}!"
    print(message)

    if 'portfolio' in sh[username]:
        print('Nice portfolio!')
    else:
        sh[username] = {
            'portfolio': pb.default_portfolio
            }
        print('You need to invest, homie!', pb.default_portfolio)
        # sh[username]['portfolio'] = pb.default_portfolio

    print(f"{sh[username]['portfolio']}")
    sh.sync()
    return username

if __name__ == "__main__":
    username = main('The Four Headless Horsemen')
    print(f"Goodbye, {username}.")