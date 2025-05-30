import torch 
from torchview import draw_graph

model = torch.nn.Sequential(
            torch.nn.Linear(8, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3)
        )

draw_graph(model, input_size=(1,8)).visual_graph.render("nn_diagram")