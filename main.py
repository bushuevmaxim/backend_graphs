from typing import List
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
from fastapi.responses import FileResponse
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib 
app = FastAPI()
matplotlib.use('agg')
@app.get("/")
def read_root():
    return {"Hello": "World"}
path_fig = "fig.png"
path_paint = "paint_gif.png"
@app.post("/show_graph")
def show_graph(graph: List[List[int]]):
    G = nx.DiGraph(np.array(graph))
    nx.draw_circular(G, node_color='brown',font_color = "whitesmoke", node_size=1000,font_size=22, with_labels=True)
    plt.show(block=False)
    plt.savefig(path_fig, format="PNG")
    return FileResponse(path=path_fig)

@app.post("/paint_graph")
def paint_graph(graph: List[List[int]]):
    G = nx.DiGraph(np.array(graph))
    nx.draw_circular(G, node_color='brown',font_color = "whitesmoke", node_size=1000,font_size=22, with_labels=True)
    plt.show(block=False)
    plt.savefig(path_paint, format="PNG")
    return FileResponse(path=path_paint)

def paint_graph(graph:List[List[int]]):
    pass
    
