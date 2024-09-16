#cell 1
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import multiprocessing as mp
import time
import heapq
import itertools


# load the faces from the norway data (calc)
def load_files(vfilename, efilename, ffilename):
    return np.loadtxt(vfilename), np.loadtxt(efilename, dtype='int'), np.loadtxt(ffilename, dtype='int')

# initialize the vertex dictionary
def initialize_vertex_dict(vertices, edges, faces):
    vertex_dict = {i: {'coords': vertices[i], 'faces': [], 'edges': []} for i in range(vertices.shape[0])}
    
    # add faces to dictionary
    for face in faces:
        for vertex_index in face:
            vertex_dict[vertex_index]['faces'].append(face)
    
    # Add edges to dictionary with consistent ordering
    for edge in edges:
        v1, v2 = edge
        
        vertex_dict[v1]['edges'].append((v1, v2))
        vertex_dict[v2]['edges'].append((v1, v2))
    
    return vertex_dict

# inital q for each vertex
def compute_initial_q(vertex_dict):
    q_matrix = {vertex_id: np.zeros((4, 4)) for vertex_id in vertex_dict.keys()}
    #print("THE DICTONARY",vertex_dict)
    
    for vertex, data in vertex_dict.items():
        #print("VERTEX",vertex)
        Q_sum = np.zeros((4, 4))
        for face in data['faces']:
            #print("FACE",face)
            p1 = vertex_dict[face[0]]['coords']
            p2 = vertex_dict[face[1]]['coords']
            p3 = vertex_dict[face[2]]['coords']
            
            vectorA = p2 - p1
            vectorB = p3 - p1
            #print("THIS IS VECTA",vectorA)
            #print("THIS IS VECTB",vectorB)
            a, b, c = np.cross(vectorA, vectorB)
            #print("a",a,"b",b,"c",c)
            d = -np.dot([a, b, c], p1)
            #print("d",d)

            Kp = np.array([[a*a, a*b, a*c, a*d],
                           [a*b, b*b, b*c, b*d],
                           [a*c, b*c, c*c, c*d],
                           [a*d, b*d, c*d, d*d]])

            Q_sum += Kp

        q_matrix[vertex] = Q_sum
    #print("THE Q MATRIX",q_matrix)
    
    return q_matrix

#compute error cost for edge contraction
def compute_error_cost(edges,q_matrix,vertex_dict):
    cost_heap = []
    
    for v1, v2 in edges:
        Q1 = q_matrix[v1]
        Q2 = q_matrix[v2]
        Q = Q1 + Q2
        #print("ROWW",v1,v2,"Q1",Q1,"\nQ2",Q2,"\nQ",Q)
        target_position = (vertex_dict[v1]['coords'] + vertex_dict[v2]['coords']) / 2
        target_position_1 = np.append(target_position, 1)
 
        error = target_position_1.T @ Q @ target_position_1

        print(f'{v1} {v2}  error  {error}\n')
      
        cost_heap.append((error, (v1, v2), target_position))
    
    heapq.heapify(cost_heap)
    
    return cost_heap

# mesh simplify (contracts edge with least cost)
def mesh_simplify(vertex_dict, cost_heap, vertices, edges, faces):
    lowest_cost = heapq.heappop(cost_heap)
    #print("Lowest error", lowest_cost[0])

    v1, v2 = lowest_cost[1]
    target_position = lowest_cost[2]
    print("Lowest", v1, v2, "Target", target_position)
    adj_edges= vertex_dict[v2]['edges']
    adj_vertices={v for edge in adj_edges for v in edge if v!=v2 }
    #print("The adj vertices",adj_vertices)
    # takes in values already in v1 (so i can merge v2)
    v1_faces = set(tuple(face) for face in vertex_dict[v1]['faces'])
    v1_edges = set(tuple(edge) for edge in vertex_dict[v1]['edges'])              
            
            

    # merging v2 (faces and edges) with v1
    for face in vertex_dict[v2]['faces']:
        new_face = tuple(v1 if vertex == v2 else vertex for vertex in face)
        if len(set(new_face)) == 3:  
            v1_faces.add(new_face)

    for edge in vertex_dict[v2]['edges']:
        new_edge = tuple([v1 if vertex == v2 else vertex for vertex in edge])
        if new_edge[0] != new_edge[1]: 
            v1_edges.add(new_edge)

    # updating v1 (in dict)
    vertex_dict[v1]['faces'] = list(v1_faces)
    vertex_dict[v1]['edges'] = list(v1_edges)

    #updating other vertex ids what had v2 (replacing with v1)
    for vertex in adj_vertices:
        update_edges= set()
     
        for edge in vertex_dict[vertex]['edges']:
     
            if v2 in edge:
                new_edge = tuple(v1 if vertex==v2 else vertex for vertex in edge)
               
                if new_edge[0]!= new_edge[1]:
                    update_edges.add(new_edge)
            else:
                update_edges.add(tuple(edge))
        vertex_dict[vertex]['edges']= list(update_edges)
        

        update_faces= set()
        for face in vertex_dict[vertex]['faces']:
            if v2 in face:
                new_face= tuple(v1 if vertex==v2 else vertex for vertex in face)
                if len(set(new_face))==3:
                    update_faces.add(new_face)
            else:
                update_faces.add(tuple(face))
        vertex_dict[vertex]['faces']= list(update_faces)


    vertex_dict[v1]['coords'] = target_position

    if v2 in vertex_dict:
        del vertex_dict[v2]

    #recalc the matrix since egdes have been changed
    q_matrix = compute_initial_q(vertex_dict)
    #updating the vertices, edges, and faces 
    edges = update_mesh_data(vertex_dict)
    #recomp cost heap (with the new matrix and is ready to have mesh simp called again)
    error_cost_heap = compute_error_cost(edges, q_matrix, vertex_dict)
    
    return vertex_dict, error_cost_heap,edges, v1, v2

#updating vertices,edges,faces
def update_mesh_data(vertex_dict):
    #getting the coords from the dict (updating)
    #vertices = np.array([data['coords'] for data in vertex_dict.values()])
    
    # getting edges from the dict (updating)
    edges_set = set()
    for data in vertex_dict.values():
        for edge in data['edges']:
            edges_set.add(tuple(edge))
    
    edges = np.array(list(edges_set), dtype=int)
    
    # getting faces from the dict (updating)
    #faces_set = set()
    #for data in vertex_dict.values():
        #for face in data['faces']:
            #faces_set.add(tuple(face))
    
    #faces = np.array(list(faces_set), dtype=int)
    
    return edges #vertices, edges, faces


#cell 2 

vertices, edges, faces= load_files("verticestest.txt","edgestest.txt","facestest.txt")
#print(vertices)
#print("EDGES",edges)
#print("FACES",faces)

vertex_dict =initialize_vertex_dict(vertices, edges, faces)
#print("VERTEX DICT",vertex_dict[0]['faces'])

q_matrix=compute_initial_q(vertex_dict)
error_cost_heap= compute_error_cost(edges, q_matrix, vertex_dict)
#vertex_dict,vertices,edges,faces= mesh_simplify(vertex_dict,error_cost_heap,vertices,edges,faces)


#cell 3 

print(vertices)
#getting all the x values 
x = vertices[:,0]

#getting all the y values 
y= vertices[:,1]
#getting all the z values 
z= vertices[:,2]
fig=plt.figure(figsize=(8,8))
ax = fig.add_subplot()

#loading data from text file (integer values are in faces- set type to int)
#for each line
for row in faces:
    #taking in three points (a,b,c) to represent vertices of triangle
    a,b,c=row
    #setting up the different points that will be plotted
    #want to get the x y and z value of that point and store them in a variable
    p1= (x[a],y[a],z[a])
    p2= (x[b],y[b],z[b])
    p3= (x[c],y[c],z[c])


    #plotting- 
    ax.plot([p1[0],p2[0]],
            [p1[1],p2[1]],color="red")
    ax.plot([p2[0],p3[0]],[p2[1],p3[1]],color="blue")
    ax.plot([p1[0],p3[0]],[p1[1],p3[1]],color="black")



plt.show()    


#cell 4
fig=plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')

for row in faces:
    #taking in three points (a,b,c) to represent vertices of triangle
    a,b,c=row
    #setting up the different points that will be plotted
    #want to get the x y and z value of that point and store them in a variable
    p1= (x[a],y[a],z[a])
    p2= (x[b],y[b],z[b])
    p3= (x[c],y[c],z[c])


    #plotting- 
  
    # Plot edges of the face
    ax.plot([p1[0],p2[0]],
            [p1[1],p2[1]],
            [p1[2],p2[2]],color="red")
    ax.plot([p2[0],p3[0]],[p2[1],p3[1]],[p2[2],p3[2]],color="blue")
    ax.plot([p1[0],p3[0]],[p1[1],p3[1]],[p1[2],p3[2]],color="black")



plt.show()   

#cell 5

int_vertices = len(vertex_dict)
target_vertex_per= int(int_vertices * 0.4)
while len(vertex_dict) > target_vertex_per:
    vertex_dict,error_cost_heap,edges,v1,v2= mesh_simplify(vertex_dict,error_cost_heap,vertices,edges,faces)
    current_vertex_count = len(vertex_dict)
    #print("Vertcies",vertices)
    #print("FACES", faces)
    if current_vertex_count<=target_vertex_per:
        break

vertices = np.array([data['coords'] for data in vertex_dict.values()])
faces_set = set()
for data in vertex_dict.values():
    for face in data['faces']:
        faces_set.add(tuple(face))
    
faces = np.array(list(faces_set), dtype=int)

#cell 6

fig=plt.figure(figsize=(8,8))
ax = fig.add_subplot()

# using the coordinates from vertex dict
for face in faces:
    p1 = vertex_dict[face[0]]['coords']
    p2 = vertex_dict[face[1]]['coords']
    p3 = vertex_dict[face[2]]['coords']

    # Plot edges of the face
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="red")
    ax.plot([p2[0], p3[0]], [p2[1], p3[1]], color="blue")
    ax.plot([p1[0], p3[0]], [p1[1], p3[1]], color="black")
    
