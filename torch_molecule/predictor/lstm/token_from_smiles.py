#Dictionary mapping each character to a corresponding token (integer).
char_dic = {
    '<pad>':0,
    '#': 1,  # Triple bond
    '%': 2,  # Two-digit ring closure (e.g., "%10")
    '(': 3,  # Branch opening
    ')': 4,  # Branch closing
    '*': 5,  # Wildcard atom (used in BigSMILES for polymer repeating units)
    '+': 6,  # Positive charge
    '-': 7,  # Negative charge
    '0': 8,  # Ring closure digit
    '1': 9, 
    '2': 10, 
    '3': 11, 
    '4': 12, 
    '5': 13, 
    '6': 14, 
    '7': 15, 
    '8': 16, 
    '9': 17, 
    '=': 18,  # Double bond
    'B': 19,  # Boron
    'C': 20,  # Carbon
    'F': 21,  # Fluorine
    'G': 22,  
    'H': 23,  # Hydrogen
    'I': 24,  # Iodine
    'K': 25,  
    'L': 26,  
    'N': 27,  # Nitrogen
    'O': 28,  # Oxygen
    'P': 29,  # Phosphorus
    'S': 30,  # Sulfur
    'T': 31,  
    'Z': 32,  
    '[': 33,  # Open bracket for isotopes, charges, or explicit atoms
    ']': 34,  # Close bracket
    'a': 35,  # Aromatic atoms
    'b': 36,  
    'c': 37,  # Aromatic carbon
    'd': 38,  
    'e': 39,  
    'i': 40,  
    'l': 41,  
    'n': 42,  # Aromatic nitrogen
    'o': 43,  # Aromatic oxygen
    'r': 44,  
    's': 45,  # Aromatic sulfur
    '/': 46,  # Cis/trans stereochemistry
    '\\': 47, # Cis/trans stereochemistry
    '@': 48,  # Chirality
    '.': 49,  # Disconnected structures
    '{': 50,  # BigSMILES / CurlySMILES polymer notation
    '}': 51,  # BigSMILES / CurlySMILES polymer notation
    '<': 52,  # CurlySMILES syntax for polymer representations
    '>': 53   # CurlySMILES syntax for polymer representations
}

def create_tensor_dataset(string_list, input_len, pad_token=0):
    """
    Converts a list of strings into tokenized sequences, pads each sequence to input_len, 
    and wraps them in a TensorDataset.
    
    Args:
        string_list (list of str): List of input strings.
        input_len (int): The fixed length to pad/truncate each token sequence.
        pad_token (int, optional): The token used for padding. Defaults to 0.

    Returns:
        List of tokens.
    """
    tokenized_list = []

    for s in string_list:
        # Convert each character in the string to a token
        tokens = [char_dic.get(char, pad_token) for char in s]
        # Pad the token sequence if it's shorter than input_len; otherwise, truncate it
        if len(tokens) < input_len:
            tokens = tokens + [pad_token] * (input_len - len(tokens))
        else:
            tokens = tokens[:input_len]

        tokenized_list.append(tokens)

    return tokenized_list
