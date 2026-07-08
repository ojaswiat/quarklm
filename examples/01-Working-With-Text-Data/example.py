# Import necessary modules
import os
import argparse
import tiktoken
from gensim.models import Word2Vec

# Define constants for the script
SEPARATOR = "=" * 64 # Define a separator line for better readability in debug output.
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
        print(SEPARATOR)
        print("""
        1. Download the file from https://en.wikisource.org/wiki/The_Verdict
        2. Save it as 'the_verdict.txt' in the 'data' folder
        3. Ensure the 'data' folder is in the same directory as this script
        """)
        print(SEPARATOR)
        raise FileNotFoundError(f"Training file not found at {file_path}. Please ensure the file exists.")
    else:
        print(SEPARATOR)
        print(f"The file contains the text of 'The Verdict' by Agatha Christie downloaded from: https://en.wikisource.org/wiki/The_Verdict")
        print(f"Training file found at {file_path}.")
        print(f"Proceeding with the tokenisation...")
        print(SEPARATOR)
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

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

        print(f"Training data loaded from {file_path}.")
        print(f"Length of training data: {len(text)} characters.")
        print(f"First 500 characters of the training data:\n{text[:500]}")
        print(SEPARATOR)

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

    if DEBUG:
        print(f"Tokenization complete. Number of tokens: {len(tokens)}")
        print(f"First 10 tokens: {', '.join(tokens[:10])}")
        print(SEPARATOR)

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

    print(f"Word2Vec model trained with {len(model.wv)} unique tokens.")
    print(f"Vector size: {model.vector_size}, Window size: {model.window}, Epochs: {model.epochs}")
    print(SEPARATOR)

    return model

# Generate a pipeline that runs the entire process: load data, tokenize, and generate embeddings
def pipeline():
    """
    Run the entire pipeline: load training data, tokenize it, and generate word embeddings.
    """

    try:
        check_training_file_exists(TRAINING_FILE_PATH)
        loaded_text = load_training_data(TRAINING_FILE_PATH)
        tokens = tokenize_text(loaded_text)
        generate_word_embeddings(tokens)
        print("Pipeline completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

# Main entry point of the script
if __name__ == "__main__":
    parse_args()
    
    print()
    if DEBUG:
        print("Debug mode is ON")
    else:
        print("Running in normal mode")

    pipeline()
    print()