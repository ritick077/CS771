import numpy as np
import matplotlib.pyplot as plt

def dist(x, u):
    d = np.zeros((x.shape[0], u.shape[0]))
    for i in range(u.shape[0]):
        diff = x - u[i, :].reshape((1, -1))
        sum_of_squares = np.zeros(x.shape[0])
        for j in range(x.shape[0]):
            row_sum = np.sum(np.square(diff[j, :]))
            sum_of_squares[j] = row_sum
        d[:, i] = sum_of_squares
    
    return d

def mean(x, c):
    u = np.zeros((2, x.shape[1]))
    for cluster_index in range(2):
        cluster_indices = np.where(c == cluster_index)[0]
        if len(cluster_indices) > 0:
            sum_values = np.zeros(x.shape[1])
            for row_index in cluster_indices:
                sum_values += x[row_index, :]
            mean_values = sum_values / len(cluster_indices)
            u[cluster_index, :] = mean_values
    return u

def cluster():
    x = np.genfromtxt('data/kmeans_data.txt', delimiter='  ')

    for iter in range(10):
        z = (np.random.randint(250, size=1)).reshape(())
        temp_y = x[z,:]
        fx = np.exp(-0.1*np.sum(np.square(x - temp_y.reshape((1,-1))), axis=1)).reshape(-1,1)

        num_rows = 2
        num_columns = fx.shape[1]
        u = np.zeros((num_rows, num_columns))
        for i in range(num_rows):
            for j in range(num_columns):
                u[i, j] = 0
        for i in range(num_rows):
            for j in range(num_columns):
                u[i, j] = fx[i, j]
        
        d1 = dist(fx , u)
        num_rows, num_columns = d1.shape
        c1 = np.zeros(num_rows, dtype=int)
        for i in range(num_rows):
            min_value = d1[i, 0]
            min_index = 0
            for j in range(1, num_columns):
                if d1[i, j] < min_value:
                    min_value = d1[i, j]
                    min_index = j
            c1[i] = min_index
        num_elements = c1.size
        c = np.zeros((num_elements, 1))
        for i in range(num_elements):
            c[i, 0] = c1[i]
         
        u = mean(fx, c)
        d2 = dist(fx , u)
        c2 = np.argmin(d2 , axis = 1)
        num_elements = c2.size
        c = np.zeros((num_elements, 1))
        for i in range(num_elements):
            c[i, 0] = c2[i]
        
        p = (c==1).reshape(c.shape[0])
        n = (c==0).reshape(c.shape[0])

        plt.figure(iter)
        plt.scatter(x[p,0], x[p,1], c='b')
        plt.scatter(x[n,0], x[n,1], c='g')
        plt.plot(x[z,0], x[z,1], 'r*')

cluster()
plt.show()
