def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

text = "red"
colored_text = colored(0, 200, 0, text)

print(colored_text)