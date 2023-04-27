import torch
import torch.nn as nn
import torch.optim as optim

# Read the input text file
filename = "input.txt"
with open(filename, "r") as f:
    text = f.read()

# Create character set and related mappings
chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

# Convert a text string to a tensor of character indices
def text_to_tensor(text, char_to_idx):
    indices = [char_to_idx[char] for char in text]
    return torch.tensor(indices)

# Define the LSTM model for text generation
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden

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

        input_seq = text_to_tensor(text[i:i+seq_length], char_to_idx).to(device)
        target_seq = text_to_tensor(text[i+1:i+seq_length+1], char_to_idx).to(device)

        predictions, hidden = model(input_seq.unsqueeze(0), hidden)
        hidden = (hidden[0].detach(), hidden[1].detach())

        loss = criterion(predictions.squeeze(0), target_seq)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / (len(text) // seq_length):.4f}")

# Function to generate text using the trained model
def generate_text(model, seed, length, temperature=1.0):
    model.eval()
    input_seq = text_to_tensor(seed, char_to_idx).to(device)
    hidden = None
    generated_text = seed

    for _ in range(length):
        predictions, hidden = model(input_seq.unsqueeze(0), hidden)
        probabilities = nn.functional.softmax(predictions.squeeze(0) / temperature, dim=-1)
        next_char_idx = torch.multinomial(probabilities[-1], 1).item()
        next_char = idx_to_char[next_char_idx]
        generated_text += next_char

        input_seq = input_seq[1:]
        input_seq = torch.cat((input_seq, torch.tensor([next_char_idx]).to(device)))

    return generated_text

# Interactive loop to generate text based on user input
while True:
    seed = input("Enter a lyrics title: ")

    generated_text = generate_text(model, seed=seed, length=600, temperature=0.4)
    print(generated_text)
