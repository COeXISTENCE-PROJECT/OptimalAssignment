import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    """
    Prosta warstwa konwolucyjna na grafie (Graph Convolution).
    Wykonuje mnożenie macierzy cech węzłów przez macierz sąsiedztwa.
    """
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        """
        x: cechy wejściowe o wymiarach (batch_size, num_channels, num_nodes, time_steps)
        A: macierz sąsiedztwa (num_nodes, num_nodes)
        
        Operacja einsum 'ncvl,vw->ncwl':
        n: batch size
        c: channels
        v: nodes (węzły źródłowe)
        l: time steps (lub inna cecha)
        w: nodes (węzły docelowe)
        
        Efektywnie mnoży cechy każdego węzła przez wagi krawędzi do jego sąsiadów.
        """
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    """
    Warstwa liniowa zrealizowana jako konwolucja 1x1.
    """
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    """
    Graph Convolution Network (GCN) module.
    Implementuje dyfuzyjną konwolucję na grafie.
    """
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        # Obliczenie wejściowej liczby kanałów dla warstwy liniowej.
        # (order * support_len + 1) wynika z tego, że bierzemy pod uwagę:
        # - oryginalny sygnał (+1)
        # - sygnały po k krokach dyfuzji (order) dla każdej macierzy wsparcia (support_len)
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        """
        x: dane wejściowe
        support: lista macierzy sąsiedztwa (np. oryginalna, transponowana, adaptacyjna)
        """
        out = [x]
        # Dla każdej macierzy wsparcia (np. graf skierowany w przód, w tył)
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            # Dyfuzja wielokrokowa (rzędu k)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        # Konkatenacja wszystkich przetworzonych sygnałów wzdłuż wymiaru kanałów
        h = torch.cat(out,dim=1)
        # Przejście przez warstwę liniową (redukcja wymiarowości i mieszanie cech)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    """
    Główna klasa modelu Graph WaveNet.
    Łączy w sobie temporal convolution (TCN) do przechwytywania zależności czasowych
    oraz konwolucje grafowe (GCN) do przechwytywania zależności przestrzennych.
    """
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        # Początkowa konwolucja 1x1 dopasowująca liczbę kanałów wejściowych do residual_channels
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        # Inicjalizacja adaptacyjnej macierzy sąsiedztwa
        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                # Losowa inicjalizacja wektorów węzłów (node embeddings)
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                # Inicjalizacja na podstawie SVD (jeśli podano aptinit)
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1


        # Budowanie warstw WaveNet (TCN)
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions (konwolucje z dylatacją)
                # Dwie ścieżki: filter i gate (mechanizm bramkowania jak w LSTM/GRU)
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=(1,new_dilation)))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=(1,new_dilation)))

                # 1x1 convolution for residual connection (połączenie rezydualne)
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection (połączenie skrótowe do wyjścia)
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                
                # Dodanie warstwy grafowej (GCN) w każdym bloku, jeśli włączone
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))


        # Warstwy wyjściowe
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, input):
        in_len = input.size(3)
        # Padding wejścia, jeśli sekwencja jest krótsza niż pole recepcyjne
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # Obliczenie adaptacyjnej macierzy sąsiedztwa raz na iterację
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            # adp = softmax(ReLU(E1 * E2))
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # Pętla przez warstwy WaveNet
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution (czasowa)
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)             #czemu tutaj tanh
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate # Gated TCN

            # parametrized skip connection (gromadzenie wyników pośrednich)
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            # Przetwarzanie przestrzenne (GCN)
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            # Dodanie połączenia rezydualnego
            x = x + residual[:, :, :, -x.size(3):]

            # Normalizacja
            x = self.bn[i](x)

        # Przetwarzanie końcowe zebranych połączeń skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
