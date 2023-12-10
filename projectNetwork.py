import numpy as np
import random

# Define the matrix for the project network
matrix1 = np.matrix('-1 4 -1 -1; -1 -1 7 10; -1 -1 -1 1; -1 -1 -1 -1')
matrix2 = np.matrix('-1 6 3 7 -1 -1 -1 -1 -1 ; -1 -1 -1 -1 5 -1 -1 -1 -1 ; -1 -1 -1 -1 2 9 -1 -1 -1 ; -1 -1 -1 -1 -1 3 -1 -1 -1 ; -1 -1 -1 -1 -1 -1 6 -1 -1 ; -1 -1 -1 -1 -1 -1 -1 1 -1 ; -1 -1 -1 -1 -1 -1 -1 -1 8 ; -1 -1 -1 -1 -1 -1 -1 -1 4 ; -1 -1 -1 -1 -1 -1 -1 -1 -1')
matrix3 = np.matrix('-1 4 2 3 -1 -1 -1 -1 -1  ; -1 -1 -1 -1 0 -1 7 -1 -1 ; -1 -1 -1 -1 0 -1 -1 0 -1  ; -1 -1 -1 -1 0 3 -1 -1 -1  ; -1 -1 -1 -1 -1 -1 -1 -1 2  ; -1 -1 -1 -1 -1 -1 0 0 -1  ; -1 -1 -1 -1 -1 -1 -1 -1 3  ; -1 -1 -1 -1 -1 -1 -1 -1 8  ; -1 -1 -1 -1 -1 -1 -1 -1 -1')
matrix12 = [[-1,4,-1,-1],[-1,-1,7,10],[-1,-1,-1,1],[-1,-1,-1,-1]]
matrix4 = np.array([[-1,20,-1,-1,-1,-1,-1,-1,-1],[-1,-1,30,60,-1,-1,-1,-1,-1],[-1,-1,-1,-1,0,-1,90,-1,-1],[-1,-1,-1,-1,0,-1,-1,80,-1],[-1,-1,-1,-1,-1,30,-1,-1,-1],[-1,-1,-1,-1,-1,-1,0,45,-1],[-1,-1,-1,-1,-1,-1,-1,30,-1],[-1,-1,-1,-1,-1,-1,-1,-1,20],[-1,-1,-1,-1,-1,-1,-1,-1,-1]])

# function to make our code read all input eg. np.matrix, np.array, 2d python array
def change_input(a):
    """
    Takes a matrix in any form and turns it into the np.matrix form

    Keyword arguments:
    a -- the matrix for the project network in either np.matrix, np.array or normal matrix form

    """
    return np.matrix(np.array(a))

def cal_EET(matrix):
    """
    Calculates the Earliest Event Time (EET) for each node
    
    Keyword arguments:
    matrix -- the matrix for the project network
    
    """
    matrix = change_input(matrix)
    num_nodes = len(matrix) 
    eet = np.full(num_nodes, 0, dtype=int) # Initialize the EET array with -1
    for node in range(num_nodes):
        if node == 0 :
            eet[node] = 0  # The starting node has EET 0
        else:
            # Calculate EET for the current node based on the maximum EET of its predecessors
            eet[node] = max(eet[i] + matrix[i, node] for i in range(num_nodes) if matrix[i, node] >= 0)
    return eet

# example usage for cal_EET
print('The Earliest Event Time of this project network is ', cal_EET(matrix1))
print('The Earliest Event Time of this project network is ', cal_EET(matrix2))
print('The Earliest Event Time of this project network is ', cal_EET(matrix3))
print('The Earliest Event Time of this project network is ', cal_EET(matrix4))

def cal_LET(matrix):
    """
    Calculates the Latest Event Time (LET) for each node
    
    Keyword arguments:
    matrix -- the matrix for the project network
    
    """
    matrix = change_input(matrix)
    num_nodes = len(matrix)
    let = np.full(num_nodes, 0, dtype=int)  # Initialize the LET array with infinity
    eet = cal_EET(matrix)
    for node in reversed(range(num_nodes)):
        if node == num_nodes - 1:
            let[node] = eet[node]  # The ending node has LET equal to its EET
        else:
            # Calculate LET for the current node based on the minimum LET of its successors
            let[node] = min(let[j] - matrix[node,j]for j in range(num_nodes) if matrix[node,j] >= 0)
    return let

# example usage for cal_LET
print('The Latest Event Time of this project network is ', cal_LET(matrix1))
print('The Latest Event Time of this project network is ', cal_LET(matrix2))
print('The Latest Event Time of this project network is ', cal_LET(matrix3))
print('The Latest Event Time of this project network is ', cal_LET(matrix4))

def cal_TF(matrix):
    """
    Calculates the Total Float (TF) for each activity
    
    Keyword arguments:
    matrix -- the matrix for the project network
    
    """
    matrix = change_input(matrix)
    num_nodes = len(matrix)
    eet = cal_EET(matrix)
    let = cal_LET(matrix)
    tf = np.zeros_like(matrix, dtype=int)  # Initialize the TF matrix with zeros
    for i in range(num_nodes):
        for j in range(num_nodes):
            if matrix[i, j] >= 0:
                # Calculate TF for each task based on LET and EET
                tf[i, j] = let[j] - eet[i] - matrix[i, j]
            else:
                tf[i, j] = -1  # No edge between tasks, set TF to -1
    return tf

# example usage for cal_TF
print('The Total Float of this project network is \n', cal_TF(matrix1))
print('The Total Float of this project network is \n', cal_TF(matrix2))
print('The Total Float of this project network is \n', cal_TF(matrix3))

def cal_CPM(matrix):
    """
    Determines the critical path by using the Critical Path Method (CPM)
    
    Keyword arguments:
    matrix -- the matrix for the project network
    
    """
    
    msg= "There are {0} critical path(s) in this network, "
    
    tf = cal_TF(matrix)
    num_nodes = len(tf)
    critical_path = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if tf[i, j] == 0:
                # Identify activities on the critical path
                critical_path.append([i + 1, j + 1])  # Adjust indices to start from 1
                
    answerList = []

    #if there is multiple path, seperate them into different lists in the correct order
    while critical_path:
        tmpList = []
        count = 0
        for i in critical_path:
            if not tmpList:
                tmpList.append(i)
                count += 1
            else:
                if i[0] == tmpList[count - 1][1]:
                    tmpList.append(i)
                    count += 1
        
        for i in tmpList:
            critical_path.remove(i)
            
        answerList.append(tmpList)

    # Reformats the print message
    number = len(answerList)
    for i in range(0,number):
        if (i == 0):
            msg = msg + str(answerList[i])
        else:
            msg = msg + ' and ' + str(answerList[i])
    print(msg.format(number))

# example usage for cal_CPM
cal_CPM(matrix1)
cal_CPM(matrix2)
cal_CPM(matrix3)
cal_CPM(matrix4)

def generate_random_project(total_nodes, max_in_degree):
    """
    Generates a random project network

    Keyword arguments:
    total_nodes -- the total amount of nodes for the project network
    max_in_degree -- the maximum in-degree for each node

    """
    # create an empty numpy array of size total_nodes by total_nodes
    random_adj_matrix = np.zeros((total_nodes, total_nodes), dtype=int)
    # change all nodes so that they dont connect with any others
    for i in range(total_nodes):
        for j in range( total_nodes):
            random_adj_matrix[i][j] = -1
    # for each node, randomly choose its predecessors from the previous nodes
    for i in range(total_nodes):
        # the number of predecessors is between 0 and max_in_degree
        num_predecessors = random.randint(0, max_in_degree)
        # the predecessors are randomly selected from the nodes with smaller index
        list = np.arange(0, total_nodes, 1).tolist()
        predecessors = random.sample(list, num_predecessors)
        # assign a random weight between 0 and 10 to each edge
        for p in predecessors:
            random_adj_matrix[p][i] = random.randint(0, 10)
    # return the random adjacency matrix
    return random_adj_matrix

## example usage for generate_random_project
print('Random project network with 30 nodes and max_in_degree 4 is \n ', generate_random_project(30,4))
