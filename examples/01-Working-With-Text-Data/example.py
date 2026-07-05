# Import necessary modules
import os
import argparse
import tiktoken
from gensim.models import Word2Vec

# Define constants for the script
DEBUG = False # Set the default value of DEBUG to False. It can be overridden by command-line arguments.
TRAINING_FILE_PATH = os.path.join(os.path.dirname(__file__), "data", "the_verdict.txt") # Specify the path to the training file.
TIKTOKEN_MODEL_NAME = "cl100k_base"  # Specify the tiktoken model name.
MAX_VECTORIZER_EPOCHS = 1 # Set the number of epochs for the Word2Vec model training. A low value is used for simplicity and speed in this example. Higher values yield better embeddings but will take longer to train.

# Parse the command-line arguments to check if debug mode is enabled
def parse_args():
    global DEBUG
    
    parser = argparse.ArgumentParser(description="Script with debug mode.")
    
    # store_true sets the value to True if the flag is present, False otherwise
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Update the global variable
    DEBUG = args.debug

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
    if not os.path.exists(file_path):
        print()
        print("""
        1. Download the file from https://en.wikisource.org/wiki/The_Verdict
        2. Save it as 'the_verdict.txt' in the 'data' folder
        3. Ensure the 'data' folder is in the same directory as this script
        """)
        raise FileNotFoundError(f"Training file not found at {file_path}. Please ensure the file exists.")
    else:
        print()
        print(f"The file contains the text of 'The Verdict' by Agatha Christie downloaded from: https://en.wikisource.org/wiki/The_Verdict")
        print(f"Training file found at {file_path}.")
        print(f"Proceeding with the tokenisation...")
        print()
        return True

# Load the training data from the specified file path
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
        text = file.read()
        return text

# Tokenize the text into words using tiktoken
def tokenize_text(text:str)->list:
    """
    Tokenize the given text into tiktoken subword tokens.

    args:
        text (str): The text to be tokenized.

    returns:
        list: A list of subword tokens obtained from tokenizing the text.
    """
    tokenizer = tiktoken.get_encoding(TIKTOKEN_MODEL_NAME)
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
    return tokens

# Generate word embeddings using Word2Vec
def generate_word_embeddings(tokens:list)->Word2Vec:
    """
    Generate word embeddings using Word2Vec from the list of tokens.

    args:
        tokens (list): A list of words obtained from tokenizing the text.

    returns:
        Word2Vec: A trained Word2Vec model containing the word embeddings.
    """
    # Prepare data for Word2Vec: Word2Vec expects a list of sentences, where each sentence is a list of words.
    # Here, we treat the entire text as a single sentence for simplicity.
    sentences = [tokens]
    
    # Train the Word2Vec model
    model = Word2Vec(
        sentences=sentences,            # Use a single sentence made from the token list
        vector_size=4,                  # Sets a 4-dimensional embedding space
        window=2,                       # Look at 2 words to the left and right - Sliding Window Size
        min_count=1,                    # Don't ignore any words
        epochs=MAX_VECTORIZER_EPOCHS,   # Number of iterations over the corpus - set to 1 for simplicity and speed
        seed=42                         # For reproducible random initialization
    )

    return model

# Main entry point of the script
if __name__ == "__main__":
    parse_args()
    
    if DEBUG:
        print("Debug mode is ON")
    else:
        print("Running in normal mode")