#Dictionary mapping each character to a corresponding token (integer).
char_dic = {
    '#': 0,  # Triple bond
    '%': 1,  # Two-digit ring closure (e.g., "%10")
    '(': 2,  # Branch opening
    ')': 3,  # Branch closing
    '*': 4,  # Wildcard atom (used in BigSMILES for polymer repeating units)
    '+': 5,  # Positive charge
    '-': 6,  # Negative charge
    '0': 7,  # Ring closure digit
    '1': 8, 
    '2': 9, 
    '3': 10, 
    '4': 11, 
    '5': 12, 
    '6': 13, 
    '7': 14, 
    '8': 15, 
    '9': 16, 
    '=': 17,  # Double bond
    'B': 18,  # Boron
    'C': 19,  # Carbon
    'F': 20,  # Fluorine
    'G': 21,  
    'H': 22,  # Hydrogen
    'I': 23,  # Iodine
    'K': 24,  
    'L': 25,  
    'N': 26,  # Nitrogen
    'O': 27,  # Oxygen
    'P': 28,  # Phosphorus
    'S': 29,  # Sulfur
    'T': 30,  
    'Z': 31,  
    '[': 32,  # Open bracket for isotopes, charges, or explicit atoms
    ']': 33,  # Close bracket
    'a': 34,  # Aromatic atoms
    'b': 35,  
    'c': 36,  # Aromatic carbon
    'd': 37,  
    'e': 38,  
    'i': 39,  
    'l': 40,  
    'n': 41,  # Aromatic nitrogen
    'o': 42,  # Aromatic oxygen
    'r': 43,  
    's': 44,  # Aromatic sulfur
    '/': 45,  # Cis/trans stereochemistry
    '\\': 46, # Cis/trans stereochemistry
    '@': 47,  # Chirality
    '.': 48,  # Disconnected structures
    '{': 49,  # BigSMILES / CurlySMILES polymer notation
    '}': 50,  # BigSMILES / CurlySMILES polymer notation
    '<': 51,  # CurlySMILES syntax for polymer representations
    '>': 52   # CurlySMILES syntax for polymer representations
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
        tokens = [char_dic[char] for char in s]
        # Pad the token sequence if it's shorter than input_len; otherwise, truncate it
        if len(tokens) < input_len:
            tokens = tokens + [pad_token] * (input_len - len(tokens))
        else:
            tokens = tokens[:input_len]
        tokenized_list.append(tokens)

    return tokenized_list
