from colorama import Fore, Style, init

init(autoreset=True)
_RESET = Style.RESET_ALL


def log_info(message: str) -> None:
    print(f"{Fore.GREEN}[INFO] {message}{_RESET}")


def log_warning(message: str) -> None:
    print(f"{Fore.YELLOW}[WARNING] {message}{_RESET}")


def log_error(message: str) -> None:
    print(f"{Fore.RED}[ERROR] {message}{_RESET}")


def log_debug(message: str) -> None:
    print(f"{Fore.BLUE}[DEBUG] {message}{_RESET}")


def log_success(message: str) -> None:
    print(f"{Fore.CYAN}[SUCCESS] {message}{_RESET}")


def log_failure(message: str) -> None:
    print(f"{Fore.MAGENTA}[FAILURE] {message}{_RESET}")


def log_critical(message: str) -> None:
    print(f"{Fore.RED}{Style.BRIGHT}[CRITICAL] {message}{_RESET}")
