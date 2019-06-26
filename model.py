import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Embeddings Layer: converts entries into a vector of a specified size
        
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer: returns the hidden states, taking the embeddings as an input
        
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)
        
        # Linear Layer : maps the dimenssion of the hidden states to the size of the vocabulary
        
        self.linear = nn.Linear(hidden_size,vocab_size)
    
    def forward(self, features, captions):
        
        # each entry within the captions is converted into an embedding, while removing the <end> entry 
        
        embeddings_vector = self.embeddings(captions[:,:-1])
        
        # stack the embeddings/features
        
        inputs = torch.cat((features.unsqueeze(1),embeddings_vector), 1)
        
        # extract the outputs and the hidden states of the LSTM layer
        
        lstm_outputs, self.hidden = self.lstm(inputs)
        
        # return the output of the fully connected layer
        
        scores = self.linear(lstm_outputs)
        
        return scores


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        sentences = []
        
        # Randomly initialize the hidden state
        
        hidden = (torch.randn(1, 1, self.hidden_size).to(inputs.device),torch.randn(1, 1, self.hidden_size).to(inputs.device))
        
        # Steps through the sequence one element at a time. After each step, 'hidden' holds the hidden state
    
        for i in range(max_len):
            out, hidden = self.lstm(inputs, hidden)
            out = self.linear(out.squeeze(1))
            tensors = out.argmax(1)
            sentences.append(tensors.item())
            
            inputs = self.embeddings(tensors.unsqueeze(0))
        
        return sentences
        
        