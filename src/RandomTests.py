import Tools as t
import AlgNaive as an
import AlgShapleyValFinalTest as alg
import AlgNaiveMoreThanK as anmt
import AlgShapleySimplified as algs
import AlgShapleyDiff as algd
'Test data tuple D_t Label'
L_t = 0

'Number of unit'
N = 3

'K from KNN classifier'
K = 2

'Bound on the fork sets size for the randomly generated data set'
forkSetLengthLowerBound = 2
forkSetLengthUpperBound = 2

'Bound on the distance to D_t for the randomly generated data set'
minDistance = 1
maxDistance= 10000

'''
User defined list of units we want the Shapley value of'
The units indices start by 1 (not by 0)
'''
unitsIndices = N*[0]
for i in range(N):
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

print("--------- Dataset Unsorted ---------")
for i in range(len(D)):
    print("D[",i,"]=",D[i])
print("")

'We sort every unit set'
for i in range(N):
    D[i] = t.sort(D[i])
    
print("--------- Dataset Sorted ---------")
for i in range(len(D)):
    print("D[",i,"]=",D[i])
print("")
    
'''
We create the sorted full pipeline output
We also store in which unit set each tuple is
'''
Dp = []
for i in range(N):
    temp = []
    for j in range(len(D[i])):
        temp.append(D[i][j]+[i+1])
    Dp = Dp+temp
Dp = t.sort(Dp)
print(Dp)

resultsAlg = []
resultsNaive = []
    
print("--------- O(M^2NK^5) version with loop optimisation --------")
algd.SP(N, K, L_t, Dp, D, unitsIndices, resultsAlg)
print(" ")
print("--------- Naive Version --------")
anmt.SP(N, K, L_t, D, unitsIndices, resultsNaive)
