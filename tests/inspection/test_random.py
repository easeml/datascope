'''
This code aims to demonstrate the correctness of our algorithm by
comparing its results with the results of the brute force approach. 
It can run the algorithm for many different values of N (number of 
unit in the pipeline) and K. It then checks that both results do not
differ by more than the given tolerance

Here the considered D_t tuple is implicit. With directly simulate the
distance from each tuple to D_t by a real number. Tuples basically of
the form [distance, label] and then for the purpose of the algorithm
[distance, label, unitIndex], where unitIndex is the
index of the unit the tuple belongs to.
'''
import tools as t
import random as r
import pytest
import alg_naive as algn
import alg_shapley as algs

# Define the range for N we want to test and the number of repetition
# for each pair (N, K)
n_lower_bound = 2
n_upper_bound = 7
n_rep = 1

# Define range of each fork set's length
fork_set_length_lower_bound = 2
fork_set_length_upper_bound = 5

# Range of the distance from D_t
min_distance = 1
max_distance = 10000

# Largest assignable label
largest_label = 2

# Define how close to the exact result our algortihm's result
# should be in order to pass a test
tol = 1e-08

# The test will check for correctness only the Shapley values
# in this array. For example, [1] will check the first unit's
# Shapley value
units_indices = [1]

parameters = []
for n in range(n_lower_bound, n_upper_bound+1):
    for k in range(2, n):
        for rep in range(n_rep):
            parameters.append([n, k, rep, r.randint(0, 1e08)])
            

@pytest.mark.parametrize("n,k,n_rep,seed", parameters)
def test_random(n, k, n_rep, seed):    
    # Test data tuple D_t Label
    L_t = 0
    
    # Number of unit
    N = n
    
    # K from KNN classifier
    K = k

    # Generate a random data set
    # This data set consists of N sets of length between forkSetLengthLowerBound and forkSetLengthLowerBound included
    # Each tuple in each of these N sets is a pair [d, l] where d represents the distance to D_t and l the label of
    # this tuple. For any d, we have minDistance <= d <= maxDistance. We also have l in {0, 1, ..., largestLabel}
    # The seed makes sure the experiment can be reproduced
    D = t.random_data_set(N, fork_set_length_lower_bound, fork_set_length_upper_bound,
                            min_distance, max_distance, largest_label, seed)
            
    # We sort every unit set
    for i in range(N):
        D[i] = t.sort(D[i])
        
    
    # We create the sorted (by increasing distance to D_t) full pipeline output
    # We also store in which unit set each tuple is
    Dp = []
    for i in range(N):
        temp = []
        for j in range(len(D[i])):
            temp.append(D[i][j]+[i+1])
        Dp = Dp+temp
    Dp = t.sort(Dp)
    
    # Will store the Shapley values of each queried unit
    results_alg = []
    results_naive = []
    
    # Run the Datascope and the brute-force version
    # print("--------- O(M^2NK^5) Version --------")
    algs.sp(N, K, L_t, Dp, D, units_indices, results_alg)
    # print("--------- Brute-Force Version --------")
    algn.sp(N, K, L_t, D, units_indices, results_naive)
    
    # Compare the results for both approaches
    for i in range(len(units_indices)):
        assert(results_alg[i]-results_naive[i] < tol)
