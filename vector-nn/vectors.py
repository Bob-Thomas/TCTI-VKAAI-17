import numpy as np


array = np.array
dot = np.dot

u = array([-4,1,2])
v = array([0,9,-6])
w = array([3,-2,-1])
x = array([1,5])
y = array([0,-8])

# 5.1

# a
#        -4 +  0   -4
#u + v =  1 +  9 = 10
#         2 + -6   -4

print("5.1.a:",u + v)

# b
#        -4 -  3   -7
#u + w =  1 - -2 =  3
#         2 - -1    3

print("5.1.b:",u - w)

# c
#       2 *  0     0
# 2v =  2 *  9 =  18
#       2 * -6   -12

print("5.1.c:",2*v)

# d
#               (3 * -4) - (2 *  0) +  3    -9
#3u - 2v + w =  (3 *  1) - (2 *  9) + -2 = -17
#               (3 *  2) - (2 * -6) + -1    17

print("5.1.d:",3 * u - 2 * v + w)

# e
#
#x + y - y = x
#

print("5.1.e:",x + y - y)

# f is ongeldig om dat je een vector van 2 groot bij een vector van 3 probeert op te stellent

#5.2

#a
# 〈u|v〉 = -4 * 0 + 1 * 9 + 2 * -6 = -3

print("5.2.a:",dot(u,v))

#b
#〈v|u〉 = 0 * -4 + 9 * 1 + -6  * 2 = -3

print("5.2.b:",dot(v,u))

#c
#〈w|x〉 = kan niet want ongelijke dimensies -6 zou met niets vemenigvuldigd worden

#d
#                   -3 *  3   -9
#〈u|v〉w = = -3 * w = -3 * -2 =  6
#                   -3 * -1    3

print("5.2.d:",dot(u,v) * w)

#e
#〈〈u|v〉w|w〉 = -9 * 3 + 6 * -2 + 3 * -1 = -42

print("5.2.e:",dot(dot(u,v) * w, w))

#f
#〈〈x|y〉w|w〉
#g
#〈〈x|y〉x|w〉 kan niet want x | w heeft ongelijke demensies
