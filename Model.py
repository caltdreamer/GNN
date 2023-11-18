# -*- coding: utf-8 -*-
"""Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12JBUAj5y69szGjubYUcnC9DeMHRttMkT
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn import BatchNorm1d as BatchNorm
import arc
from arc import models
from arc import methods
from arc import black_boxes
from arc import others
from arc import coverage
class SAGERegressor(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
            super(SAGERegressor, self).__init__()

            self.num_layers = num_layers
            self.convs = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))

            for i in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                self.batch_norms.append(BatchNorm(hidden_channels))

            self.convs.append(SAGEConv(hidden_channels, out_channels))
            self.batch_norms.append(BatchNorm(out_channels))
            self.linear1 = torch.nn.Linear(out_channels, hidden_channels)
            self.linear2 = torch.nn.Linear(hidden_channels, 1)

        def reset_parameters(self):

            for conv in self.convs:
                conv.reset_parameters()
            for bn in self.batch_norms:
                bn.reset_parameters()
            self.linear1.reset_parameters()
            self.linear2.reset_parameters()

        def forward(self, x, adjs):
            for i, (edge_index, _, size) in enumerate(adjs):
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)

                x = self.batch_norms[i](x)
                if i != self.num_layers - 1:
                    x = F.leaky_relu(x, 0.2)
                    x = F.dropout(x, p=0.1, training=self.training)

            x = F.leaky_relu(self.linear1(x), 0.2)
            x = self.linear2(x)

            return x

class SAGE(torch.nn.Module):
      def __init__(self,in_channels,hidden_channels,out_channels,num_layers = 3):
          super(SAGE,self).__init__()

          self.numlayers = num_layers
          self.convs = torch.nn.ModuleList()
          self.convs.append(SAGEConv(in_channels,hidden_channels))
          for i in range(num_layers - 2):
              self.convs.append(SAGEConv(hidden_channels,hidden_channels))
          self.convs.append(SAGEConv(hidden_channels,out_channels))
      def reset_parameters(self):
          for conv in self.convs:
              conv.reset_parameters()
      def forward(self,x,adjs):
          for i ,(edge_index,_,size) in enumerate(adjs):
              xs = []
              x_target = x[:size[1]]#x should be a list prioritizing target nodes
              x = self.convs[i]((x,x_target),edge_index)
              if i != self.numlayers -1:
                  x = F.relu(x)
                  x = F.dropout(x,p=0.5,training=self.training)
              xs.append(x)
              if i == 0:
                  x_all = torch.cat(xs, dim=0)
                  layer_1_embeddings = x_all
              elif i == 1:
                  x_all = torch.cat(xs, dim=0)
                  layer_2_embeddings = x_all
              elif i == 2:
                  x_all = torch.cat(xs, dim=0)
                  layer_3_embeddings = x_all
          return layer_1_embeddings, layer_2_embeddings, layer_3_embeddings