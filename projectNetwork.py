import numpy as np
import random

# Define the matrix for the project network
matrix1 = np.matrix('-1 4 -1 -1; -1 -1 7 10; -1 -1 -1 1; -1 -1 -1 -1')
matrix2 = np.matrix('-1 6 3 7 -1 -1 -1 -1 -1 ; -1 -1 -1 -1 5 -1 -1 -1 -1 ; -1 -1 -1 -1 2 9 -1 -1 -1 ; -1 -1 -1 -1 -1 3 -1 -1 -1 ; -1 -1 -1 -1 -1 -1 6 -1 -1 ; -1 -1 -1 -1 -1 -1 -1 1 -1 ; -1 -1 -1 -1 -1 -1 -1 -1 8 ; -1 -1 -1 -1 -1 -1 -1 -1 4 ; -1 -1 -1 -1 -1 -1 -1 -1 -1')
matrix3 = np.matrix('-1 4 2 3 -1 -1 -1 -1 -1  ; -1 -1 -1 -1 0 -1 7 -1 -1 ; -1 -1 -1 -1 0 -1 -1 0 -1  ; -1 -1 -1 -1 0 3 -1 -1 -1  ; -1 -1 -1 -1 -1 -1 -1 -1 2  ; -1 -1 -1 -1 -1 -1 0 0 -1  ; -1 -1 -1 -1 -1 -1 -1 -1 3  ; -1 -1 -1 -1 -1 -1 -1 -1 8  ; -1 -1 -1 -1 -1 -1 -1 -1 -1')

def cal_EET(matrix):
    """
    Calculates the Earliest Event Time (EET) for each node
    
    Keyword arguments:
    matrix -- the matrix for the project network
    
    """
    num_nodes = len(matrix) 
    eet = np.full(num_nodes, 0, dtype=int) # Initialize the EET array with -1
    for node in range(num_nodes):
        if node == 0 :
            eet[node] = 0  # The starting node has EET 0
        else:
            # Calculate EET for the current node based on the maximum EET of its predecessors
            eet[node] = max(eet[i] + matrix[i, node] for i in range(num_nodes) if matrix[i, node] >= 0)
    return eet

def cal_LET(matrix):
    """
    Calculates the Latest Event Time (LET) for each node
    
    Keyword arguments:
    matrix -- the matrix for the project network
    
    """
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

def cal_TF(matrix):
    """
    Calculates the Total Float (TF) for each activity
    
    Keyword arguments:
    matrix -- the matrix for the project network
    
    """
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

def cal_CPM(matrix):
    """
    Determines the critical path by using the Critical Path Method (CPM)
    
    Keyword arguments:
    matrix -- the matrix for the project network
    
    """
    
    msg= "The Critical Path of this project network is {0}"
    
    tf = cal_TF(matrix)
    num_nodes = len(tf)
    critical_path = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if tf[i, j] == 0:
                # Identify activities on the critical path
                critical_path.append([i + 1, j + 1])  # Adjust indices to start from 1
                
    answerList = []
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
    print(msg.format(answerList))
    
def generate_random_project(total_nodes, max_in_degree):
    # create an empty numpy array of size total_nodes by total_nodes
    random_adj_matrix = np.zeros((total_nodes, total_nodes))
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
            if random_adj_matrix[i][p] == -1:
                random_adj_matrix[p][i] = random.randint(0, 10)
            list.remove(p)
    # return the random adjacency matrix
    print(random_adj_matrix)
    return random_adj_matrix
