import numpy as np

array = np.array
matrix = np.matrix
dot = np.dot


A = matrix('0 -1; 0 1')
B = matrix([[-3, 0],[0,2]])
u = array([3,5])
v = array([8,2])


#a
#Au = [[0 , -1], * [ 3 = [0 * 3 + -1 * 5, 0 * 3 + 5 * 1] = [-5, 5]
#     [0,   1]]      5 ]

print('5.3.a:',dot(A, u))

#b
#B(Au) =

print(dot(B, dot(A, u).T))
