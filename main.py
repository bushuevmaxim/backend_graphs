import base64
import io
from typing import List
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
from fastapi.responses import FileResponse
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib 
import random
import math
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
matplotlib.use('agg')
path_fig = "fig.png"
path_paint = "paint_gif.png"
@app.post("/show_graph")
def show_graph(graph: List[List[int]]):
    G = nx.Graph(np.matrix(graph), create_using=nx.Graph)
    nx.draw_circular(G, node_color='brown',font_color = "whitesmoke", node_size=1000,font_size=22, with_labels=True)
    plt.savefig(path_fig, format="png")
    plt.clf()
    with open(path_fig, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read())
    data = {"img": b64_string}
    json_data = jsonable_encoder(data)
    return JSONResponse(content=json_data)
@app.post("/test")
def test(graph: List[List[int]]):
    G = nx.Graph(np.matrix(graph), create_using=nx.Graph)
    nx.draw_circular(G, node_color='brown',font_color = "whitesmoke", node_size=1000,font_size=22, with_labels=True)
    plt.savefig(path_fig, format="PNG")
    plt.clf()
    with open(path_fig, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read())
    data = {"img": b64_string, "count": {4}}
    json_data = jsonable_encoder(data)
    return JSONResponse(content=json_data)
@app.post("/paint_graph")
def paint_graph(graph: List[List[int]], temperature: float, cooling_rate: float):
    G = nx.Graph(np.matrix(graph), create_using=nx.Graph)
    colorlist = list(paint_graph(graph,temperature, cooling_rate ))
    nx.draw_circular(G, node_color=colorlist,font_color = "black", node_size=1000,font_size=22, with_labels=True)
    plt.savefig(path_paint, format="PNG")
    plt.clf()
    with open(path_paint, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read())
    
    data = {"img": b64_string, "count": str(len(set(colorlist)))}
    json_data = jsonable_encoder(data)
    return JSONResponse(content=json_data)

def paint_graph(graph:List[List[int]], initial_temperature, cooling_rate):
    return simulated_annealing(graph, initial_temperature, cooling_rate)
    

def simulated_annealing(adjacency_matrix, initial_temperature, cooling_rate):
    num_nodes = len(adjacency_matrix)
    current_solution = initialize_solution(num_nodes)  
    current_cost = calculate_cost(adjacency_matrix, current_solution) 
    temperature = initial_temperature

    while temperature > 0.1:  
        for i in range(100): 
            new_solution = generate_neighbor_solution(current_solution)
            new_cost = calculate_cost(adjacency_matrix, new_solution) 

            if new_cost < current_cost:
                current_solution = new_solution
                current_cost = new_cost
            num_colors_used = len(set(current_solution))
            if num_colors_used == num_nodes and current_cost == 0:
                return current_solution
        temperature *= cooling_rate

    return current_solution

def initialize_solution(num_nodes):
    solution = [-1] * num_nodes
    return solution

def calculate_cost(adjacency_matrix, solution):
    cost = 0
    num_nodes = len(adjacency_matrix)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i][j] == 1 and solution[i] == solution[j]:
                cost += 1
    return cost

def generate_neighbor_solution(solution):
    new_solution = solution.copy()
    node = random.randint(0, len(solution) - 1)
    new_solution[node] = random.randint(0, max(solution) + 1)
    return new_solution