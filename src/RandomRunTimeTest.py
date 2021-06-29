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
import Tools as t
import AlgShapleySimplified as algs
import time

Ns = [15]

'Set range of the test'
nLowerBound = 3
nUpperBound = 10


nRepEachCase = 1
'Allowed difference between the brute-force and the Datascope approach'
tol = 1e-08



for n in Ns:
    for k in range(2, n):
        timee = 0.0
        for r in range(nRepEachCase):
            'Test data tuple D_t Label'
            L_t = 0
            
            'Number of unit'
            N = n
            
            'K from KNN classifier'
            K = k
            
            'Bound on the fork sets size for the randomly generated data set'
            forkSetLengthLowerBound = n
            forkSetLengthUpperBound = n
            
            'Bound on the distance to D_t for the randomly generated data set'
            minDistance = 1
            maxDistance= 1000000
            
            '''
            User defined list of units we want the Shapley value of'
            The units indices start by 1 (not by 0)
            '''
            unitsIndices = 1*[0]
            for i in range(1):
                unitsIndices[i] = i+1
            
            'Largest label'
            largestLabel = 2
            
            '''
            Generate a random data set
            This data set consists of N sets of length between forkSetLengthLowerBound and forkSetLengthLowerBound included
            Each tuple in each of these N sets is a pair [d, l] where d represents the distance to D_t and l the label of
            this tuple. For any d, we have minDistance <= d <= maxDistance. We also have l in {0, 1, ..., largestLabel}
            '''
            D = t.randomDataSet(N, forkSetLengthLowerBound, forkSetLengthUpperBound, minDistance, maxDistance, largestLabel)
                    
            'We sort every unit set'
            for i in range(N):
                D[i] = t.sort(D[i])
                
            '''
            We create the sorted (by increasing distance to D_t) full pipeline output
            We also store in which unit set each tuple is
            '''
            Dp = []
            for i in range(N):
                temp = []
                for j in range(len(D[i])):
                    temp.append(D[i][j]+[i+1])
                Dp = Dp+temp
            Dp = t.sort(Dp)
            
            'Will store the Shapley values of each queried unit'
            resultsAlg = []
            resultsNaive = []
            
            'Run the Datascope and the brute-force version'
            startTime = time.time()
            algs.SP(N, K, L_t, Dp, D, unitsIndices, resultsAlg)
            executionTime = (time.time() - startTime)
            timee = timee+executionTime
                
        print("Case N =", n, " and K =", k)
        print("Time needed on average: ", timee/nRepEachCase)
        print("")