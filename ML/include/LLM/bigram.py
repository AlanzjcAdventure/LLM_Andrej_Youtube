import torch
import torch.nn.functional as F

words = open("names.txt", "r").read().splitlines()
vocal_size = len(words)

# Prepare the datasets
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

block_sz = 1 # context size to predict next char, 1 for bigram
x, y = [], []

for word in words:
    context = [0] * block_sz
    for ch in word + '.':
        ix = stoi[ch]
        x.append(context)
        y.append(ix)
        #print([itos[i] for i in context], "predict next word", itos[ix])
        context = context[1:] + [ix]

# Split the datasets
train_sz = int(0.9 * vocal_size)
xtr, ytr = x[:train_sz], y[:train_sz]
xvar, yvar = x[train_sz:], y[train_sz:]

# Train the model
X = torch.tensor(xtr)
Y = torch.tensor(ytr)

g = torch.Generator()
g.manual_seed(2147483647)
W = torch.randn((27, 27), generator = g)
b = torch.randn((27), generator = g)
max_steps = 20000
lr = 0.5
reg_param = 0.5

parameters = [W, b]
print("number of elements:", sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

num = X.nelement()
print("number of examples:", num)

for i in range(max_steps):
    # forward pass
    # encode input as one_hot vector
    xenc = F.one_hot(X, num_classes=27).float()
    logits = xenc @ W + b

    # counts = logits.exp()
    # logits_norm = counts/counts.sum(1,keepdim=True)
    # loss = -logits_norm[torch.arange(num), Y].log().mean() + reg_param*(W**2).mean()
    loss = F.cross_entropy(logits.squeeze(), Y)

    # backward pass
    # set the grad to zero so it won't accumulate previous values
    for p in parameters:
        p.grad = None
    loss.backward()

    # update the parameters
    for p in parameters:
        p.data += - lr*p.grad

    if i%1000 == 0:
        print(f'{i}/{max_steps}:', loss.item())
    


# validate the performance
with torch.no_grad():
    Xvar = torch.tensor(xvar)
    Yvar = torch.tensor(yvar)
    xenc = F.one_hot(Xvar, num_classes = 27).float()
    logits = xenc @ W + b
    var_loss = F.cross_entropy(logits.squeeze(),Yvar)
    print("validation loss:", var_loss.item())

# sample the output using multinominal distribution
for _ in range(5):
    ix = 0
    output = []
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes = 27).float()
        logits = xenc @ W
        counts = logits.exp() 
        p = counts / counts.sum(1, keepdim=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        output.append(itos[ix])
        if ix==0:
            break
    print(''.join(output))

# trainning with 20000 times, i think we can go forward with more steps to reduce underfitting issue
# Training loss: 2.32
# validation loss: 2.56 