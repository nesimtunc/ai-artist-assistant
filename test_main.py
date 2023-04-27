import torch
from text_generator import TextGenerator
from utils import text_to_tensor, generate_text

def test_text_to_tensor():
    char_to_idx = {'a': 0, 'b': 1, 'c': 2}
    text = "abc"
    expected_tensor = torch.tensor([0, 1, 2])

    result = text_to_tensor(text, char_to_idx)
    assert torch.equal(result, expected_tensor), f"Expected {expected_tensor}, but got {result}"

def test_generate_text():
    vocab_size = 3
    embedding_dim = 2
    hidden_dim = 2

    model = TextGenerator(vocab_size, embedding_dim, hidden_dim)
    model.eval()  # Set model to evaluation mode for testing

    # Set up fixed weights to make sure the test is deterministic
    for param in model.parameters():
        param.data.fill_(0.5)

    seed = "a"
    length = 5
    temperature = 1.0
    char_to_idx = {'a': 0, 'b': 1, 'c': 2}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    result = generate_text(model, seed, length, temperature, char_to_idx, idx_to_char)

    # Check if the generated text has the correct length
    assert len(result) == length + len(seed), f"Expected a string of length {length + len(seed)}, but got {len(result)}"

    # Check if the generated text contains only valid characters
    for char in result:
        assert char in char_to_idx, f"Generated text contains an invalid character: {char}"
