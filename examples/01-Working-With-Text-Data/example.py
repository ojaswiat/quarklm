# Import necessary libraries
import os
import tiktoken
from gensim.models import Word2Vec

# Set the Debug mode to True to see the intermediate outputs and debug information
DEBUG_MODE = True

TRAINING_FILE_PATH = os.path.join(os.path.dirname(__file__), "assets", "the_verdict.txt")

# Check if the training file exists
def check_training_file_exists(file_path)->bool:
    """
    Check if the training file exists at the specified path.
    If the file does not exist, print instructions for downloading it and raise a FileNotFoundError.

    args:
        file_path (str): The path to the training file.
    raises:
        FileNotFoundError: If the training file does not exist at the specified path.
    """
    if not os.path.exists(TRAINING_FILE_PATH):
        print()
        print("""
        1. Download the file from https://en.wikisource.org/wiki/The_Verdict
        2. Save it as 'the_verdict.txt' in the 'assets' folder
        3. Ensure the 'assets' folder is in the same directory as this script
        """)
        raise FileNotFoundError(f"Training file not found at {TRAINING_FILE_PATH}. Please ensure the file exists.")
    else:
        print()
        print(f"The file contains the text of 'The Verdict' by Agatha Christie downloaded from: https://en.wikisource.org/wiki/The_Verdict")
        print(f"Training file found at {TRAINING_FILE_PATH}.")
        print(f"Proceeding with the tokenisation...")
        print()
        return True

def load_training_data(file_path)->str:
    """
    Load the training data from the specified file path.

    args:
        file_path (str): The path to the training file.

    returns:
        str: The content of the training file.
    """
    
    check_training_file_exists(file_path)

    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def tokenize_text(text:str)->list:
    """
    Tokenize the given text into words using tiktoken.

    args:
        text (str): The text to be tokenized.

    returns:
        list: A list of words obtained from tokenizing the text.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_ids = tokenizer.encode(text)
    words = [tokenizer.decode([token_id]) for token_id in token_ids]
    return words

# Read the content of the training file
with open(TRAINING_FILE_PATH, 'r', encoding='utf-8') as file:
    text = file.read()

    # Create a tokenizer using tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Tokenize the text into words
    token_ids = tokenizer.encode(text)

    # Convert token IDs back to words
    words = [tokenizer.decode([token_id]) for token_id in token_ids]
