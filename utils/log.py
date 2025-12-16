from colorama import Fore, Style, init

def log_info(message: str) -> None:
    init(autoreset=True)
    print(f"{Fore.GREEN}[INFO] {message}{Style.RESET_ALL}")

def log_warning(message: str) -> None:
    init(autoreset=True)
    print(f"{Fore.YELLOW}[WARNING] {message}{Style.RESET_ALL}") 

def log_error(message: str) -> None:
    init(autoreset=True)
    print(f"{Fore.RED}[ERROR] {message}{Style.RESET_ALL}")  

def log_debug(message: str) -> None:
    init(autoreset=True)
    print(f"{Fore.BLUE}[DEBUG] {message}{Style.RESET_ALL}") 

def log_success(message: str) -> None:  
    init(autoreset=True)
    print(f"{Fore.CYAN}[SUCCESS] {message}{Style.RESET_ALL}")

def log_failure(message: str) -> None:      
    init(autoreset=True)
    print(f"{Fore.MAGENTA}[FAILURE] {message}{Style.RESET_ALL}")

def log_critical(message: str) -> None:      
    init(autoreset=True)
    print(f"{Fore.RED}{Style.BRIGHT}[CRITICAL] {message}{Style.RESET_ALL}")