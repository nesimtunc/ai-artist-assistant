import torch
import torch.nn as nn
import torch.optim as optim
from text_generator import TextGenerator
from utils import text_to_tensor, generate_text

# Read the input text file
filename = "input.txt"
with open(filename, "r") as f:
    text = f.read()

# Create character set and related mappings
chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

# Instantiate the model, loss function, and optimizer
embedding_dim = 164
hidden_dim = 256

model = TextGenerator(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
num_epochs = 150
seq_length = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    hidden = None

    for i in range(0, len(text) - seq_length, seq_length):
        optimizer.zero_grad()

        input_seq = text_to_tensor(
            text[i:i+seq_length], char_to_idx).to(device)
        target_seq = text_to_tensor(
            text[i+1:i+seq_length+1], char_to_idx).to(device)

        predictions, hidden = model(input_seq.unsqueeze(0), hidden)
        hidden = (hidden[0].detach(), hidden[1].detach())

        loss = criterion(predictions.squeeze(0), target_seq)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / (len(text) // seq_length):.4f}")


# Interactive loop to generate text based on user input
while True:
    seed = input("Enter a lyrics title: ")

    generated_text = generate_text(model,
                                   seed=seed,
                                   length=600,
                                   temperature=0.4,
                                   char_to_idx=char_to_idx,
                                   idx_to_char=idx_to_char)
    print(generated_text)
