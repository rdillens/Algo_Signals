import shelve
import questionary
import portfolio_builder as pb

def main(display_string=None):
    if display_string is not None:
        print(display_string)

    # Get username
    username = questionary.text("What is your name?").ask()

    # Open shelf to access persistent data
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
        # Check to see if portfolio exists in shelf
        if 'portfolio' in sh[username]:
            print('Nice portfolio!', sh[username]['portfolio'])
            if not questionary.confirm("Do you want to keep it?").ask():
                sh[username] = pb.init_portfolio(sh[username])
                sh.sync()
        # If portfolio does not exist, prompt user to create one
        else:
            print('You need to invest, homie!')
            sh[username] = pb.init_portfolio(sh[username])
            sh.sync()
        
        # Check to see if balance exists in shelf
        if 'balance' in sh[username]:
            if sh[username]['balance'] > 0.0:
                print(f"Looking good ${sh[username]['balance']}!")
            else:
                print(f"Current balance ${sh[username]['balance']}.")
            if questionary.confirm("Do you want to add funds?").ask():
                sh[username] = pb.add_funds(sh[username])
                sh.sync()
        # If balaance does not exist, prompt user for input
        else:
            print(f"balance not in shelf for {username}")
            sh[username] = pb.add_funds(sh[username])
            sh.sync()

        print(f"{sh[username]['portfolio']}\nBalance: ${sh[username]['balance']}")
    return username

if __name__ == "__main__":
    username = main('The Four Headless Horsemen')
    print(f"Goodbye, {username}.")