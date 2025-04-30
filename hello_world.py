import sys

print("Hello, World!")
print("python version: ", sys.version)
name = input("Enter your name: ")
print("Hello, " + name + "!")
from thumbor.utils import logger

# Then use these in your code:
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error(f"Error during face recognition: {str("dd")}")
logger.critical("This is a critical error")