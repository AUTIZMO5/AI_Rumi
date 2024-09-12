import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from torch.cuda.amp import GradScaler, autocast

batch_size = 16
block_size = 256
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256  # Reduced
n_head = 4    # Reduced
n_layer = 4   # Reduced
dropout = 0.2


torch.manual_seed(1337)

#rumi dataset
with open('input.txt','r',encoding='utf-8')as f:
  text=f.read()

#all sorted chars
chars = sorted(list(set(text)))
vocab_size = len(chars)

#encode and decode chars to ints
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #char -> int
decode = lambda l: ''.join([itos[i] for i in l]) #int -> char

data = torch.tensor(encode(text), dtype=torch.long)
# split up the data into train and validation sets
n = int(0.9*len(data)) #90% train 10% val
train_data = data[:n]
val_data = data[n:]

#make predictions/data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
   out = {}
   model.eval()
   for split in ['train','val']:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
         X,Y = get_batch(split)
         logits,loss = model(X,Y)
         losses[k] = loss.item()
      out[split] = losses.mean()
   model.train()
   return out 

class Head(nn.Module):
   """ one head of self-attention """

   def __init__(self, head_size):
      super().__init__()
      self.key = nn.Linear(n_embd, head_size, bias=False)
      self.query = nn.Linear(n_embd, head_size, bias=False)
      self.value = nn.Linear(n_embd, head_size, bias=False)
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

      self.dropout = nn.Dropout(dropout)  

   def forward(self, x):
      B,T,C = x.shape
      k = self.key(x)
      q = self.query(x)

      wei = q @ k.transpose(-2,-1) * C**-0.5
      wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
      wei = F.softmax(wei, dim=-1)
      wei = self.dropout(wei)  

      v = self.value(x)
      out = wei @ v
      return out
   
class MultiHeadAttention(nn.Module):
   """ multiple heads of self-attention """
   def __init__(self, head_size, num_heads):
      super().__init__()
      self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
      self.proj = nn.Linear(n_embd, n_embd)
      self.dropout = nn.Dropout(dropout)

   def forward(self, x):
      out = torch.cat([h(x) for h in self.heads], dim=-1)
      out = self.dropout(self.proj(out))
      return out
   
class FeedForward(nn.Module):
   """ a simple linear layer follwed by a non-linearity """
   def __init__(self, n_embd):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(n_embd, 4 * n_embd),
         nn.ReLU(),
         nn.Linear(4 * n_embd,n_embd),
         nn.Dropout(dropout),
      )
   
   def forward(self, x):
      return self.net(x)
   
class Block(nn.Module):
   """ transformer block: communication follwed by computation """
   def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
         
   def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x
 
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape 
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
          B, T, C = logits.shape
          logits = logits.view(B*T, C)
          targets = targets.view(B*T)
          loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
print("Running on:", device)  # Ensure it says 'cuda' if you have a GPU

model = BigramLanguageModel()
m = model.to(device)

#train the model
# Initialize the mixed precision scaler
scaler = torch.amp.GradScaler('cuda')

# Your existing variables
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

total_time = 0  # Initialize a variable to accumulate the time taken for all iterations

for iter in range(max_iters):
    iteration_start_time = time.time()  # Start the timer at the beginning of the loop

    if iter % eval_interval == 0:
        eval_start_time = time.time()  # Start timer for evaluation
        losses = estimate_loss()  # Evaluate and get the loss
        eval_end_time = time.time()  # End timer for evaluation
        
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Add the evaluation time to the total time
        total_time += (eval_end_time - eval_start_time)
        print(f"Total evaluation time for step {iter}: {eval_end_time - eval_start_time:.2f} seconds")
        print(f"Total accumulated time up to step {iter}: {total_time:.2f} seconds")

    # Get a batch of training data
    xb, yb = get_batch('train')

    optimizer.zero_grad(set_to_none=True)
    
    # Forward pass and backward pass with mixed precision
    with torch.amp.autocast('cuda'):
        logits, loss = model(xb, yb)
    
    # Scale the loss to prevent underflow
    scaler.scale(loss).backward()
    
    # Update model parameters
    scaler.step(optimizer)
    
    # Update the scaler for the next iteration
    scaler.update()
    
    # Clear the cache to free up GPU memory
    torch.cuda.empty_cache()

    # End of iteration, update the total time
    iteration_end_time = time.time()
    total_time += (iteration_end_time - iteration_start_time)
    print(f"Iteration {iter} took {iteration_end_time - iteration_start_time:.2f} seconds")
    print(f"Total accumulated time after iteration {iter}: {total_time:.2f} seconds")

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))





