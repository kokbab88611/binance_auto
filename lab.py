def my_function(*args):
    for arg in args:
        print(arg)

user_input = input("Enter arguments (separated by spaces): ")
arguments = user_input.split()

my_function(*arguments)
