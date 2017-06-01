import numpy as np

cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
#cost = np.array([[]])
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(cost)

from munkres import Munkres
munkres = Munkres()

print(row_ind,col_ind)
print ("printed assignements")
print (cost)
print ("printed cost matrix")

assignements = munkres.compute(cost)
print (assignements)
print ("printed assignements")
print (cost)
print ("printed cost matrix")