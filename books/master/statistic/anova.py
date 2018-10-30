# simple analysis of variance analyse the mean diff in 
# more than two group
import pandas as pd
import numpy as np

def F_score(df):
    """
         MS_between
    F = ------------
         MS_within
    """
    n = df.count().values[0]
    N = df.count().sum()
    num_of_group = df.shape[1]
    free_between = num_of_group - 1
    free_within = N - num_of_group
    sum_of_element_square_g = df.applymap(np.square).sum()
    mean_of_square_of_sum_g = df.sum() ** 2 / n
    MS_between = mean_of_square_of_sum_g.sum() - df.sum().sum() ** 2 / N
    MS_within = sum_of_element_square_g.sum() - mean_of_square_of_sum_g.sum()
    MS_all = sum_of_element_square_g.sum() - df.sum().sum() **2 / N
    F = (MS_between / free_between) / (MS_within / free_within)
    return F

if __name__ == '__main__':
    data = {
        'g1': [87, 86, 76, 56, 78, 98, 77, 66, 75, 67],
        'g2': [87, 85, 99, 85, 79, 81, 82, 78, 85, 91],
        'g3': [89, 91, 96, 87, 89, 90, 89, 96, 96, 93],
    }    
    df = pd.DataFrame(data=data, index=range(10))
    f = F_score(df)
    print("f-score %.3f" % f)

    # we have left a vital problem here
    # where the F-score table comes from?
    # by refer to a F crit table, we got
    threshold = 3.36
    if f > threshold:
        msg = "reject null hypothesis"
    else:
        msg = 'cannot reject null hypothesis'
    print(msg)
    res = "F_2_27 = %.3f, p < .05" % f
    print(res)