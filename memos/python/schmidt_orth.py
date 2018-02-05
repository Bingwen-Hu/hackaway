import numpy as np

def inner_multiple(vector1, vector2):
    return sum(v1 * v2 for (v1, v2) in zip(vector1, vector2))

def norm(vector):
    return np.sqrt(sum(v * v for v in vector))

def schmidt_orth(vectors):
    num = len(vectors)
    vectors_orth = []
    for vec in vectors:
        sub_vec = schmidt_orth_sum_helper(vec, vectors_orth)
        vec_orth = vec - sub_vec
        vectors_orth.append(vec_orth)

    # normal
    vectors_orth = np.array(vectors_orth)
    vectors_orth = np.array([v/norm(v) for v in vectors_orth])
    return vectors_orth

def schmidt_orth_sum_helper(vector, vectors_orth):
    sub_vectors = []
    for v_o in vectors_orth:
        rate = inner_multiple(v_o, vector) / inner_multiple(v_o, v_o)
        sub_vectors.append(rate * np.array(v_o))
    return np.sum(sub_vectors, axis=0)
        

        
if __name__ == "__main__":
    a1 = [1, 2, -1]
    a2 = [-1, 3, 1]
    a3 = [4, -1, 0]
    vectors = np.array([a1, a2, a3])
    vectors_orth = schmidt_orth(vectors)
    print(vectors_orth)
