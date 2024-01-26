import logging

def get_logger(name="SatClass"):
    # create logger with 'spam_application'
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)
    
    return logger

class CustomFormatter(logging.Formatter):
    BOLD_RED = "\x1b[31;1m"
    BLACK = '\033[0;30m'
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    BROWN = '\033[0;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    GREY = '\033[0;37m'
    DARK_GREY = '\033[1;30m'
    LIGHT_RED = '\033[1;31m'
    LIGHT_GREEN = '\033[1;32m'
    YELLOW = '\033[1;33m'
    LIGHT_BLUE = '\033[1;34m'
    LIGHT_PURPLE = '\033[1;35m'
    LIGHT_CYAN = '\033[1;36m'
    WHITE = '\033[1;37m'
    RESET = "\033[0m"

    format_time = "%(asctime)s - %(name)s - %(levelname)s "
    format_slash = "| "
    format_message = "%(message)s"
    # format = "%(asctime)s - %(name)s - %(levelname)s | %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: GREY + format_time + RESET + GREY + format_slash + format_message + RESET,
        logging.INFO: GREEN + format_time + RESET + GREY + format_slash + format_message + RESET,
        logging.WARNING: YELLOW + format_time + RESET + GREY + format_slash + format_message + RESET,
        logging.ERROR: RED + format_time + RESET + GREY + format_slash + format_message + RESET,
        logging.CRITICAL: BOLD_RED + format_time + RESET + GREY + format_slash + format_message + RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)