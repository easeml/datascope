import Tools as t
import time

def SP(N, K, L_t, D, unitIndices, results):   
    for l in unitIndices:
        startTime = time.time()
        phi_q = 0.0
        q = l
        combinations = []
        current = []
        'Stored all the possible binary vector of length N into combinations'
        t.allCombi(N, current, combinations)
        'Remove the combinations where the qth value is 1'
        combinations = t.filterQ(combinations, q)
        
        'For each of the binary vector of length N and vector[q] = 0:'
        for vector in combinations:
            'We compute the marginal contribution of the q-th unit:'
            'We store the output of the pipeline when the q-th unit is off'
            output = t.pipelineOutput(D, vector)
            vectorQ = []
            vectorQ = t.copyVector(vector)
            vectorQ[q-1] = 1
            'We store the output of the pipeline when the q-th unit is on'
            outputQ = t.pipelineOutput(D, vectorQ)
            'If the top K isnt filled with tuple, we skip this case'
            if(len(output) < K or len(output) < K):
                continue
            'Get the proportion of tuple with label L_t in the top K both'
            'when the q-th unit is on and off'
            propWithQ = t.topKWithLt(outputQ, K, L_t)
            propWithoutQ = t.topKWithLt(output, K, L_t)
            'Compute the marginal contribution from the soft accuracy function'
            diff = propWithQ-propWithoutQ
            # if(diff != 0):
            #     print(vector)
            vMv = diff/K
            count = t.countOnes(vector)
            vMv = vMv/t.choose(N-1, count)
            phi_q = phi_q + vMv
        phi_q = phi_q/N
        executionTime = (time.time() - startTime)
        print(q,"-th unit's shapley value", phi_q, "| Execution time:", executionTime)
        results.append(phi_q)