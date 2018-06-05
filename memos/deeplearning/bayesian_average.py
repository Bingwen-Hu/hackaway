# bayesian average
"""    12.13  12.14 all    average
下雪	33	   92	125	     0.736
那些年	139	   146	285	     0.512
李宇春	1	   4	5	     0.8
看见	145	   695	840  	 0.827
（平均）            313.75	 0.719

reduce the impact of small dataset by baysian average
"""
import numpy as np

lst = [
    [33, 92],
    [139, 146],
    [1, 4],
    [145, 695],
]

def bayesian_average(lst):
    average_vote = np.sum(lst) / len(lst)
    score_list = [l/(f+l) for (f, l) in lst]
    average_score = sum(score_list) / len(score_list)

    bayesian_score = [((f + l) * score + average_score * average_vote) / (f + l + average_vote) for score, (f, l) in zip(score_list, lst)]
    return bayesian_score

score = bayesian_average(lst)
print(score)



