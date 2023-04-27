import torch
import torch.nn.functional as F


def text_to_tensor(text, char_to_idx):
    '''
    Convert a text string to a tensor of character indices
    '''
    indices = [char_to_idx[char] for char in text]
    return torch.tensor(indices)


def generate_text(model, seed, length, temperature, char_to_idx, idx_to_char):
    '''
    Function to generate text using the trained model
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    input_seq = text_to_tensor(seed, char_to_idx).to(device)
    hidden = None
    generated_text = seed

    for _ in range(length):
        predictions, hidden = model(input_seq.unsqueeze(0), hidden)
        probabilities = F.softmax(predictions.squeeze(0) / temperature, dim=-1)
        next_char_idx = torch.multinomial(probabilities[-1], 1).item()
        next_char = idx_to_char[next_char_idx]
        generated_text += next_char

        input_seq = input_seq[1:]
        input_seq = torch.cat(
            (input_seq, torch.tensor([next_char_idx]).to(device)))

    return generated_text
