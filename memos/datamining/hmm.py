# https://blog.csdn.net/zhangduan8785/article/details/80443650

# hidden statess
states = ('health', 'sick')

# observations 
observations = ('well', 'cold', 'dizzy')

# init states probability
init_probas = {'health': 0.6, 'sick': 0.4}

# states transitions matrix
T_matrix = {
    'health': {'health': 0.7, 'sick': 0.3},
    'sick': {'health': 0.4, 'sick': 0.6},
}

# emission probability
emission_probas = {
    'health': {'well': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'sick': {'well': 0.1, 'cold': 0.3, 'dizzy': 0.6}
}

######### Questions:
# 1. well
# 2. cold
# 3. dizzy
# what the most probably states of my body?

# the very first day
f_health = init_probas['health'] * emission_probas['health']['well']
f_sick = init_probas['sick'] * emission_probas['sick']['well']
summary = f"""First day is well, P(health) = P(init_health) * P(well|health) = {f_health:0.3f},
P(sick) = P(init_sick) * P(well|sick) = {f_sick:0.3f}, because P(health) {'>' if f_health > f_sick else '<'} P(sick), 
so the first day, the my state is {'health' if f_health > f_sick else 'sick'}"""
print(summary)

# the second day
s_health = max(f_health * T_matrix['health']['health'], f_sick * T_matrix['sick']['health']) * emission_probas['health']['cold']
s_sick = max(f_health * T_matrix['health']['sick'], f_sick * T_matrix['sick']['sick']) * emission_probas['sick']['cold']
summary = f"""Second day is cold, P(health) = max{{P(f_health) * P(health->health), P(f_sick) * P(sick->health)}} * P(cold|health) = {s_health:0.3f},
P(sick) = max{{P(f_health) * P(health->sick), P(f_sick) * P(sick->sick)}} * P(cold|sick) = {s_sick:0.3f}, because P(health) {'>' if s_health > s_sick else '<'} P(sick), 
so the second day, the my state is {'health' if s_health > s_sick else 'sick'}"""
print(summary)
