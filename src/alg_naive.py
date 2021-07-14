import tools as t

def sp(n, k, L_t, d, unit_indices, results):   
    for l in unit_indices:
        phi_q = 0.0
        q = l
        combinations = []
        current = []
        # Stored all the possible binary vector of length N into combinations
        t.all_combi(n, current, combinations)
        # Remove the combinations where the qth value is 1
        combinations = t.filter_Q(combinations, q)
        
        # For each of the binary vector of length N and vector[q] = 0:
        for vector in combinations:
            # We compute the marginal contribution of the q-th unit:
            # We store the output of the pipeline when the q-th unit is off
            output = t.pipeline_output(d, vector)
            vector_q = []
            vector_q = t.copy_vector(vector)
            vector_q[q-1] = 1
            # We store the output of the pipeline when the q-th unit is on
            output_q = t.pipeline_output(d, vector_q)
            # If the top K isnt filled with tuple, we skip this case
            if(len(output) < k or len(output) < k):
                continue
            # Get the proportion of tuple with label L_t in the top K both
            # when the q-th unit is on and off'
            prop_with_q = t.topK_with_Lt(output_q, k, L_t)
            prop_without_q = t.topK_with_Lt(output, k, L_t)
            # Compute the marginal contribution from the soft accuracy function
            diff = prop_with_q-prop_without_q
            vMv = diff/k
            count = t.count_ones(vector)
            vMv = vMv/t.choose(n-1, count)
            phi_q = phi_q + vMv
        phi_q = phi_q/n
        print(q,"-th unit's shapley value", phi_q)
        results.append(phi_q)