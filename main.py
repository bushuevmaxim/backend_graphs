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

matrix = [[0,1,0,1,1,0,0,1,0,1],
     [1,0,1,0,0,1,0,0,1,0],
     [0,1,0,0,0,0,0,1,0,0],
     [1,0,0,0,1,0,1,0,0,0],
     [1,0,0,1,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0,1,0],
     [0,0,0,1,0,0,0,1,1,1],
     [1,0,1,0,0,0,1,0,1,0],
     [0,1,0,0,0,1,1,1,0,0],
     [1,0,0,0,0,0,1,0,0,0]]
app = FastAPI()
matplotlib.use('agg')
path_fig = "fig.png"
path_paint = "paint_gif.png"
@app.post("/show_graph")
def show_graph(graph: List[List[int]] = matrix):
    G = nx.DiGraph(np.array(graph))
    nx.draw_circular(G, node_color='brown',font_color = "whitesmoke", node_size=1000,font_size=22, with_labels=True)
    plt.show(block=False)
    plt.savefig(path_fig, format="PNG")
    return FileResponse(path=path_fig)

@app.post("/paint_graph")
def paint_graph(graph: List[List[int]]= matrix):
    G = nx.DiGraph(np.array(graph))
    colorlist = list(paint_graph(graph))
    print(colorlist)
    nx.draw_circular(G, node_color=colorlist,font_color = "black", node_size=1000,font_size=22, with_labels=True)
    plt.show(block=False)
    plt.savefig(path_paint, format="PNG")
    return FileResponse(path=path_paint)

def paint_graph(graph:List[List[int]]):
    return simulated_annealing(graph, temperature=100 , cooling_rate=0.95).values()
    
    

import random
import math

def initialize_colors(graph):
    """
    Initialize random colors for each node in the graph.
    Colors are represented as integers ranging from 1 to the maximum degree of the graph.
    """
    colors = {}
    num_nodes = len(graph)
    for node in range(num_nodes):
        colors[node] = random.randint(1, max_degree(graph))
    return colors

def max_degree(graph):
    """
    Calculate and return the maximum degree among all nodes in the graph.
    """
    max_degree = 0
    num_nodes = len(graph)
    for node in range(num_nodes):
        degree = sum(graph[node])
        if degree > max_degree:
            max_degree = degree
    return max_degree

def calculate_cost(graph, colors):
    """
    Calculate the total number of conflicts (edges between nodes of the same color) in the graph for a given coloring.
    """
    cost = 0
    num_nodes = len(graph)
    for node in range(num_nodes):
        node_color = colors[node]
        for neighbor in range(num_nodes):
            if graph[node][neighbor] == 1 and colors[neighbor] == node_color:
                cost += 1
    return cost

def kempe_chain_swap(colors, node_a, node_b):
    """
    Perform a Kempe chain swap between two nodes in the coloring.
    Swap the colors of nodes that have either color color_a or color_b.
    """
    color_a = colors[node_a]
    color_b = colors[node_b]
    for node in colors:
        if colors[node] == color_a:
            colors[node] = color_b
        elif colors[node] == color_b:
            colors[node] = color_a

def simulated_annealing(graph, temperature, cooling_rate):
    """
    Implement the simulated annealing algorithm to find an optimal coloring for the graph.
    """
    num_nodes = len(graph)
    current_solution = initialize_colors(graph)
    best_solution = current_solution.copy()
    current_cost = calculate_cost(graph, current_solution)
    best_cost = current_cost

    while temperature > 0.01:
        new_solution = current_solution.copy()
        node_a = random.randint(0, num_nodes - 1)
        node_b = random.randint(0, num_nodes - 1)

        kempe_chain_swap(new_solution, node_a, node_b)
        new_cost = calculate_cost(graph, new_solution)

        if new_cost < current_cost or math.exp((current_cost - new_cost) / temperature) > random.random():
            current_solution = new_solution
            current_cost = new_cost

        if new_cost < best_cost:
            best_solution = new_solution
            best_cost = new_cost

        temperature *= cooling_rate

    return best_solution