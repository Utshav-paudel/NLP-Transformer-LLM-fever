#@ Implementaton of GPT language model
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))      # for masking multi head
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # input of size (batch, timestep, channels)
        # output of size (batch, timestep, head_size)
        B,T,C = x.shape                        # unpacking the shape
        k = self.key(x)                        # B,T,hs
        q = self.query(x)                      # B,T,hs
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # (B,T,hs) @ (B,hs,T) = (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))                # (B,T,T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)                         # B,T,hs
        out =  wei @ v                            # (B,T,T) @ (B,T,hs) = (B,T,hs)
        return out
    
    
class MultiHeadAttention(nn.Module):                                                # Multihead attention : Determine attention for multiple head
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])           
        self.proj = nn.Linear(head_size * num_heads, n_embd )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)            # (B:batch,T:time,C:feature) : [h1,h1,h1,h1,h2,h2,h2,h2.....,hn,hn,hn,hn]
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):               # feedforward of decoder 
    def __init__(self, n_embd):             # linear -> RELU -> Linear
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(n_embd, 4*n_embd),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout),)
    def forward(self,x):
        return self.net(x)
        
class Block(nn.Module):
    #Transformer block : communication followed by computation
    
    def __init__(self,n_embd,n_head):  # constructor initializing decoder member item
        #n_embd : embedding dimension , n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head                    # number of feature each head is caputring
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self,x):            # forming deocder block         
        y = self.sa(x)              # multihead attention
        x = self.ln1(x+y)           # Adding and norm
        y = self.ffwd(x)            # feedforward : Linear -> RELU -> Linear
        x = self.ln2(x + y)         # Adding and norm
        return x
    

class GPTLanguageModel(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)           # token embedding 
        self.position_embedding_table = nn.Embedding(block_size, n_embd)        # positional embedding 
        
        # decoder 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for i in range(n_layer)])  # providing  decoder = n_layer sequentially
        
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)  
        self.apply(self._init_weights)
        
        
    def _init_weights(self, module):                                            # weight initialization
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)                               # set bias to zeros
        elif isinstance(module, nn.Embedding):                                  # if module is object of nn.Embedding 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)            # normal initialization of weights with less variation
    
    
    def forward(self, index, targets = None):
        B,T = index.shape
        # index and targets are both (B,T) tensor of integers
        tok_embd = self.token_embedding_table(index)       # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x= tok_embd + pos_embd                                                    # (B,T,C ) by broadcasting
        x= self.blocks(x)                                                         # decoder 
        x= self.ln_f(x)                                                           # layer normalizing
        logits= self.lm_head(x)                                                   # (B,T,vocab_size)
                                                                                   
        if targets is None:
            loss = None
        else:
            B, T, C  = logits.shape                       # B (batch size), T (sequence length), and C (embedding dimensions)
            logits = logits.view(B*T, C)                  # reshape to total word, embedding dimension
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)       # loss between predicted logits and actual logits
        return logits, loss                               # return predicted logits and loss
    
    def generate(self , index, max_new_tokens):
        # index is (B,t) array of indices in the current context
        for i in range(max_new_tokens):
            logits, loss = self.forward(index)            # get the predictions
            # focus on the last step only
            logits = logits[:,-1,:]                       # becomes (B,C)
            # apply softmax to get prob.
            probs = F.softmax(logits, dim=-1)             # (B,C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples =1) # (B,1)
            # append sampled inext to the running sequene
            index = torch.cat((index, index_next), dim=1)        # (B, T+1)
        return index
    
model = GPTLanguageModel(vocab_size)
m = model.to(device)