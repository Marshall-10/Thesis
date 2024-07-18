from colorama import Fore, Back
from colorama import init
init(autoreset=True)

import colored_traceback
colored_traceback.add_hook(always=True)

# Colors: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
#         LIGHTBLACK_EX, LIGHTRED_EX, LIGHTGREEN_EX, LIGHTYELLOW_EX, LIGHTBLUE_EX,
#         LIGHTMAGENTA_EX, LIGHTCYAN_EX, LIGHTWHITE_EX
def cprintf(fstring, color):
    """
    Colorful foreground print in terminal.
    """
    if color == 'black':
        print(Fore.BLACK + fstring)
    elif color == 'red':
        print(Fore.RED + fstring)
    elif color == 'green':
        print(Fore.GREEN + fstring)
    elif color == 'yellow':
        print(Fore.YELLOW + fstring)
    elif color == 'blue':
        print(Fore.BLUE + fstring)
    elif color == 'magenta':
        print(Fore.MAGENTA + fstring)
    elif color == 'cyan':
        print(Fore.CYAN + fstring)
    elif color == 'white':
        print(Fore.WHITE + fstring)
    elif color == 'l_black':
        print(Fore.LIGHTBLACK_EX + fstring)
    elif color == 'l_red':
        print(Fore.LIGHTRED_EX + fstring)
    elif color == 'l_green':
        print(Fore.LIGHTGREEN_EX + fstring)
    elif color == 'l_yellow':
        print(Fore.LIGHTYELLOW_EX + fstring)
    elif color == 'l_blue':
        print(Fore.LIGHTBLUE_EX + fstring)
    elif color == 'l_magenta':
        print(Fore.LIGHTMAGENTA_EX + fstring)
    elif color == 'l_cyan':
        print(Fore.LIGHTCYAN_EX + fstring)
    elif color == 'l_white':
        print(Fore.LIGHTWHITE_EX + fstring)