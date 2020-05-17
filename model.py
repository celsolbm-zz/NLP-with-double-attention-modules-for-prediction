import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from nxlearn.ml.modules import RBLSTM
from torch.nn.utils.rnn import pack_padded_sequence



class NLP_double_attention(nn.Module):
    def __init__(self, event_size=114,edge_event_size = 83, word_size = 32, d_a = 128,r1 =8,r2 = 8, event_embedding_size=128, hidden_size=256, 
                 encoding_size=64,):
        super().__init__()
        self.ws1_item = torch.nn.Linear(word_size,d_a) #WS1
        self.ws1_item.bias.data.fill_(0) #bias zerao
        self.ws2_item = torch.nn.Linear(d_a,r1) #Ws2, if using multiple attention heads
        self.ws2_item.bias.data.fill_(0)
        self.r1 = r1
        
        self.ws1_item2 = torch.nn.Linear(word_size,d_a) #WS1
        self.ws1_item2.bias.data.fill_(0) #bias zerado
        self.ws2_item2 = torch.nn.Linear(d_a,r1) #Ws2, if using multiple attention heads
        self.ws2_item2.bias.data.fill_(0)
        
        self.lstm_events = RBLSTM(event_embedding_size,hidden_size,2,p=0.02)
        self.ws1_events = torch.nn.Linear(hidden_size*2,d_a) #WS1
        self.ws1_events.bias.data.fill_(0) #bias zerado
        self.ws2_events = torch.nn.Linear(d_a,r1) #Ws2, if using multiple attention heads
        self.ws2_events.bias.data.fill_(0)
        self.r2 = r2
        self.inter_event = torch.nn.Linear(hidden_size*2,32)
        
        self.events_embedding= torch.nn.Linear(event_size,event_embedding_size)
        self.edges_embedding = torch.nn.Linear(edge_event_size,event_embedding_size)
        
        self.inter_inicio = torch.nn.Linear(event_size,event_embedding_size) #WS1
        self.activation = F.relu
        self.linear_final = torch.nn.Linear(32,1)    
        
    def masked_softmax(self,vector,mask,dim=0): #for the attention module
        for k in range(vector.size(1)):
            vec = torch.cat((torch.softmax(vector[:mask[k],k,:] ,axis = dim),torch.zeros([vector.size(0)-mask[k],self.r1]).to(device)))
            vec = vec.unsqueeze(1)
            try:
                out = torch.cat((out,vec),axis=1)
            except:
                out = vec
        return out
    def forward(self, inputs, seq_len,item2, item2_len, item, item_len):
        phrase_1 = F.relu(self.ws1_item(item))
        phrase_2 = self.ws2_item(phrase_1)
        phrase_2 = self.masked_softmax(phrase_2,item_len)
        attention_phrase = phrase_2.transpose(0,1)
        attention_phrase = attention_phrase.transpose(1,2)
        sentence_embeddings = attention_phrase@(item.transpose(0,1))
        avg_sentence_embeddings = torch.sum(sentence_embeddings,1)/self.r1
        
        item2_1 = F.relu(self.ws1_item2(item2))
        item2_2 = self.ws2_item2(item2_1)
        item2_2 = self.masked_softmax(item2_2,item2_len)
        attention_item2 = item2_2.transpose(0,1)
        attention_item2 = attention_item2.transpose(1,2)
        item2_embeddings = attention_item2@(item2.transpose(0,1))
        avg_item2_embeddings = torch.sum(item2_embeddings,1)/self.r1
               
        event = torch.cat((inputs[1:-1,:,:50].squeeze(1).float(),avg_sentence_embeddings.float(),avg_item2_embeddings.float()),axis = 1)
        event_embeds = self.events_embedding(event)
        begin_embeds1 = self.edges_embedding(inputs[0,:,:].float())
        end_embeds1 = self.edges_embedding(inputs[-1,:,:].float())
        total_embeds = torch.cat((begin_embeds1,event_embeds,end_embeds1),axis=0)
        outputs = self.lstm_events(total_embeds.unsqueeze(1).float(), seq_len)
        
        x = F.relu(self.ws1_events(outputs))
        x = self.ws2_events(x)
        x = self.masked_softmax(x,seq_len)
        
        attention = x.transpose(0,1)
        attention = attention.transpose(1,2)
        event_embeddings = attention@(outputs.transpose(0,1))
        avg_event_embeddings = torch.sum(event_embeddings,1)/self.r1
        inter_event = F.relu(self.inter_event(avg_event_embeddings))
        final = self.linear_final(inter_event)
        return final