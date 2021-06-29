import Tools as t
import AlgNaive as an
import AlgOptiAlpha as aoa
import AlgNaiveMoreThanK as anmt
import AlgShapleySimplified as algs
import AlgShapleyDiff as algd
import AlgNaiveMoreThanK as anmt

'D_t Label'
L_t = 0
'Number of unit'
N = 3
K = 2
'Used to sum all units Shapley value up'
sumVar = 0

unitsIndices = N*[0]
for i in range(N):
    unitsIndices[i] = i+1

D1 = [[1210.896, 2], [1845.669, 2]]
D2 = [[6473.669, 0], [8888.276, 1]]
D3 = [[953.757, 0], [2394.395, 1]]


D = [D1, D2, D3]

Dp = []
for i in range(N):
    temp = []
    for j in range(len(D[i])):
        temp.append(D[i][j]+[i+1])
    Dp = Dp+temp
Dp = t.sort(Dp)

resultsAlg = []
resultsNaive = []

print("--------- Optimised Version --------")
algd.SP(N, K, L_t, Dp, D, unitsIndices, resultsAlg)
print(" ")
print("--------- Naive Version --------")
anmt.SP(N, K, L_t, D, unitsIndices, resultsNaive)