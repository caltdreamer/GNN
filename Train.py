# -*- coding: utf-8 -*-
"""Train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lom2qkvvsODKYT5NToI-d6RRu5Mvqt_N
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm


def train_regression(model, data, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(data.x[n_id], adjs)
        loss = F.mse_loss(out, data.y[n_id][:batch_size].unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss
def train_classification(model,data,train_loader,optimizer,device,y,x,
                         train_mask):
      model.train()
      total_loss = total_correct =0

      for batch_size,n_id,adjs in train_loader:
          adjs = [adj.to(device) for adj in adjs]
          optimizer.zero_grad()
          l1_emb, l2_emb, l3_emb = model(x[n_id], adjs)
          out = l3_emb.log_softmax(dim=-1)
          loss = F.nll_loss(out,y[n_id[:batch_size]])
          loss.backward()
          optimizer.step()

          total_loss += float(loss)
          total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())


      loss = total_loss / len(train_loader)


      approx_acc = total_correct / int(train_mask.sum())

      return loss, approx_acc



def inference_regression(subgraph_loader,model,x):
        model.eval()
        for batch_size,n_id,adjs in subgraph_loader:

            #emb1,emb2,emb3 = model(x[n_id],adjs)
            emb3 = model(x[n_id],adjs)

            c = x[n_id][:batch_size]
            fn = torch.cat((c, emb3), dim=1)
        return emb3,fn

def inference_classification(subgraph_loader,model,x,y,train_mask):
      model.eval()
      total_loss = total_correct = 0
      for batch_size,n_id,adjs in subgraph_loader:

          emb1,emb2,emb3 = model(x[n_id],adjs)
          out = emb3.log_softmax(dim=-1)
          y_pred = out.argmax(dim=-1,keepdim =True)
          c = x[n_id][:batch_size]
          fn = torch.cat((c, emb3.softmax(dim=-1)), dim=1)
          return fn,emb3, y_pred