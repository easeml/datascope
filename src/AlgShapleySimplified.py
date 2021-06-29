import numpy as np
import Tools as t
import time

'''
This algorithm computes the Shapley value of all units in a parameterized pipeline. 
Parameters
----------
N :     integer
        the pipeline's number of unit
K :     integer
        the parameter K of the considered KNN classifier
L_t :   integer  
        the label of the test data point D_t
Dp :    set of tuple
        the whole pipeline output. Tuple in this set are of the form
        [d, l, i] where d denotes the distance of the tuple from D_t,
        l the label of the tuple and i the fork set's index the tuple
        belongs to.
D :     a set containing all the fork sets
uI :    [integer]
        the set of unit indices we want the Shapley value of
res :   [double]
        used to store the Shapley values of the unit in uI
        
We will distinguish two cases:
Case 1: D_b in in q-th unit's fork set (and D_c not) or D_b and D_c aren't in the q-th unit's
        fork set, but they are in the same fork set.
Case 2: D_b and D_c are in a different fork set that are different from the q-th one.
We distinguish between those two because in the first case we need to ensure that 2
units are turned on at the start, and in the second case, 3 need to be on at the
start, so the computation of the DP table is different.
'''
def SP(N, K, L_t, Dp, D, uI, res):
    'We loop over all requested unit indices'
    for q in uI:
        #startTime = time.time()
        phi_q = 0.0
        'We loop over every tuple pair (D_b, D_c) where D_b is the K-th nearest neighbour when'
        'unit q is on and D_c when unit q is off'
        for D_b in Dp:
            for D_c in Dp:
                'D_c can neither be part of the q-th fork set as it is the Kth nearest neighbour when'
                'unit q is off nor nearer from D_t than D_b as turning q-th unit on can only expel'
                'further tuples from the top K.'
                if(D_c[2] != q and D_c[0] >= D_b[0]):
                    'We declare the tally vectors'
                    tallies = 4*[np.array([]).tolist()]
                    'We store the unit index of tuple D_b and D_c'
                    i = D_b[2]
                    j = D_c[2]
                    
                    'For every fork set different from i, j and q having at least one tuple close'
                    'enough to D_t to have chance participating to the top K (either when the q-th'
                    'unit is on or off), we compute its corresponding tally vectors value'
                    for h in range(N):
                        if h != i-1 and h != j-1 and h != q-1 and t.tooFarSet(D[h], D_b, D_c):
                            tallies = t.copyVector(t.appendTally(tallies, D[h], D_b, D_c, L_t))
                    
                    'We add the tally vector value of unit j'        
                    tallies = t.copyVector(t.appendTally(tallies, D[j-1], D_b, D_c, L_t))
                    
                    'If there are only 2 units turned on by default we indicate it by m = 0'
                    'to the dynamic program. Otherwise we also need to add the tally values'
                    'of unit i and we indicate it by m = 1'
                    if((i == q and i != j) or (i != q and i == j)):
                        m = 0
                    else:
                        m = 1
                        tallies = t.copyVector(t.appendTally(tallies, D[i-1], D_b, D_c, L_t))
                    
                    'We add the q-th unit tally values'    
                    tallies[0].append(t.count(D[q-1], D_b, 0, L_t))
                    tallies[2].append(t.count(D[q-1], D_b, 1, L_t))
                    'We get the number of unit with tuples that may participate to the top K'    
                    L = len(tallies[0])
                    
                    'Aims to shorten the DP computation below'
                    t1, t2, t3, t4 = tallies[0], tallies[1], tallies[2], tallies[3]
                    DP = np.asarray(np.array((L+1)*[(K+1)*[(K+1)*[(K+1)*[(K+1)*[(K+1)*[0]]]]]]))
                    'Base cases'
                    for n in range(L+1):
                        DP[n][0][0][0][0][0] = 1
                    'Recurrence'
                    for n in range(1,L+1):
                        for a in range(K+1):
                            for l1 in range(K+1):
                                for l2 in range(K+1):
                                    for l11 in range(K+1):
                                        for l22 in range(K+1):
                                            'Case a) If the current unit is the q-th one, we need to turn it on and if we cannot this'
                                            '        this entry should remain 0'
                                            'Case b) If the current unit is the i-th or j-th one, we need to turn them on and if we cannot'
                                            '        this entry should remain 0'
                                            'Case c) the default DP recurrence, we consider both cases; when we turn the current unit on and'
                                            '        when we do not'
                                            # a)
                                            if(n == L):
                                                if(l1>=t1[n-1] and l11>=t3[n-1]):
                                                    DP[n][a][l1][l2][l11][l22] = DP[n-1][a][l1-t1[n-1]][l2][l11-t3[n-1]][l22]
                                            # b)
                                            elif((m == 0 and n == L-1) or (m == 1 and (n == L-1 or n == L-2))):
                                                if(a > 0):
                                                    if(l1>=t1[n-1] and l2>=t2[n-1] and l11>=t3[n-1] and l22>=t4[n-1]):
                                                        DP[n][a][l1][l2][l11][l22] = DP[n-1][a-1][l1-t1[n-1]][l2-t2[n-1]][l11-t3[n-1]][l22-t4[n-1]]
                                            # c)
                                            else:    
                                                if(a != 0):
                                                    if (l1>=t1[n-1] and l2>=t2[n-1] and l11>=t3[n-1] and l22>=t4[n-1]):
                                                        DP[n][a][l1][l2][l11][l22] = DP[n-1][a][l1][l2][l11][l22] + DP[n-1][a-1][l1-t1[n-1]][l2-t2[n-1]][l11-t3[n-1]][l22-t4[n-1]]
                                                    else:
                                                        DP[n][a][l1][l2][l11][l22] = DP[n-1][a][l1][l2][l11][l22]
                    'Number of unit that had no tuple who could participate in the top K'
                    nOutSet = N-len(t1)
                    m = 2
                    if((i == q and i != j) or (i != q and i == j)):
                        m = 1
                    'We loop over every possible proportion of tuple with label l_t'
                    'We basically apply the Shapley value formula'                    
                    for l1 in range(K+1):
                        for l2 in range(K+1):
                            DP_val = 0
                            'We first consider case where at most K units are on'
                            'a represents the number of unit on, b the number of unit among'
                            'these a units that have no tuple in the top K. We look at each'
                            'possible b, such that we have a total of a unit. For example, '
                            'if b = 2, then we look at a-2 unit on in the DP table and then'
                            'we multiply nOutSet choose 2 to simulate the activation of too'
                            'arbitrary unit without any tuple in the top K'
                            for a in range(m,K+1):
                                for b in range(nOutSet+1):
                                    if(a-b >= m):
                                        DP_val = DP_val + t.choose(nOutSet, b)*DP[L][a-b][l1][l2][K-l1][K-l2]/t.choose(N-1, a)
                            'Then we consider cases with more than K units on. The only'
                            'difference with the previous case is that b must at least be'
                            'a-K since we cannot have more than K units only in the top K'
                            for a in range(K+1, N):
                                for b in range(a-K,nOutSet+1):
                                    if(a-b >= m):
                                        DP_val = DP_val + t.choose(nOutSet, b)*DP[L][a-b][l1][l2][K-l1][K-l2]/t.choose(N-1, a)
                            phi_q = phi_q+(l1-l2)*DP_val
        phi_q = phi_q/(N*K)
        #executionTime = (time.time() - startTime)
        #print(q,"-th unit's shapley value", phi_q, "| Execution time:", executionTime)
        res.append(phi_q)
