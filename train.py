import json

from sympy import Idx

from langprocessing import tokenization,stemming,word_dir
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from module import NeuralNet

with open('intents.json','r') as f:
    intents=json.load(f)

#print(intents)

all_words=[]
tags=[]
xy=[]

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)

    for pattern in intent ['patterns']:
        w = tokenization(pattern)
        all_words.extend(w)
        xy.append((w,tag))

ignore_words= ['?','!','.',',']

all_words=[stemming(w) for w in all_words if w not in ignore_words]
all_words=sorted(set(all_words))

tags=sorted(set(tags))
'''print(tags)
print(all_words)'''


X_train=[]
Y_train=[]


for (pattern_sentence,tag) in xy:
    dir=word_dir(pattern_sentence,all_words)
    X_train.append(dir)

    label=tags.index(tag)
    Y_train.append(label)


X_train=np.array(X_train)
Y_train=np.array(Y_train)


class chatdataset (Dataset):
    def __init__(self):
        self.n_samples=len(X_train)
        self.X_data=X_train
        self.Y_data=Y_train

    def __getitem__(self,index):
        return self.X_data[Idx], self.Y_data[Idx]
    
    def __len__(self):
        return self.n_samples


batch_size=8
hidden_size =8
output_size=len(tags)
input_size=len(X_train[0])
learning_rate=0.001
num_epochs=1000


dataset=chatdataset()
train_loader=DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
module=NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(module.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for(words,labels) in train_loader:
        words=words.to(device)
        labels=labels.to(device)

        outputs=module(words)
        loss=criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if(epoch+1)%100==0:
        print(f'epoch{epoch+1}/{num_epochs},loss={loss.item():.4f}')

print(f'final loss,loss={loss.item():.4f}')

data={
    "model_state": module.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "output_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE="data.pth"
torch.save(data,FILE)
print(f"training complete.file saved to {FILE}")