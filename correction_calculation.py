import numpy as np
from scipy.stats import spearmanr
import math

def cosine_similarity(dict1, dict2):
    list1 = list(dict1.values())
    list2 = list(dict2.values())
    
    return np.dot(list1, list2) / (np.linalg.norm(list1) * np.linalg.norm(list2))

def max_difference(dict1, dict2):
    list1 = list(dict1.values())
    list2 = list(dict2.values())
    max = 0
    for i in range(len(list1)):
        if abs(list1[i]-list2[i]) > max:
            max = abs(list1[i]-list2[i])
    return max


Exact_Shapley = np.array([2, 1, 9, 7, 5, 8, 4, 3, 6, 0])

TMC_Shapley = np.array([9, 5, 8, 4, 2, 6, 1, 7, 3, 0])

MAB_Shapley = np.array( [2, 1, 9, 7, 3, 5, 8, 6, 4, 0])

GTG_Shapley = np.array([8, 6, 5, 9, 2, 1, 3, 0, 4, 7])

Delta_Shapley = np.array([8, 5, 6, 9, 0, 3, 4, 2, 1, 7])

 
# 计算斯皮尔曼等级相关性
# coef, p_value = spearmanr(Exact_Shapley, MAB_Shapley) # 0.5515151515151515
# coef, p_value = spearmanr(Exact_Shapley, GTG_Shapley) # 1
coef, p_value = spearmanr(Exact_Shapley, MAB_Shapley)
 
print(f"斯皮尔曼等级相关系数: {coef}")
# print(f"p值: {p_value}")

Exact_Shapley_Value = {0: 1.5876518306485026, 1: 0.8068055428150632, 2: 0.11137939994592994, 3: 1.279711987829162, 4: 1.1199919513737167, 5: 1.0871433701720983, 6: 1.3234647788935252, 7: 1.0793962702746438, 8: 1.1176914387840842, 9: 0.9877635206968075}

Delta_Shapley_Value = {0: 9.020998001098631, 1: 26.661592344443005, 2: 15.947496573130296, 3: 13.56062835454941, 4: 14.308996081352237, 5: -4.464767952760058, 6: -1.3978689114252736, 7: 36.53915731112163, 8: -5.385231991608935, 9: 1.4365010460217797}

TMC_Shapley_Value = {0: 2.6675816747546204,	1: 2.3853823201855024,	2: 2.4439755974213275,	3: 2.395910966495672,	4: 2.4621516346931465,	5: 3.227918646335602,	6: 2.4477353355288503,	7: 2.4507206476728123,	8: 2.266778006652991,	9: 2.3787773587306345	}

MAB_Shapley_Value = {0: 1.3503093291074038,	1: 0.6877922984957695,	2: 0.18123900060852371,	3: 1.0031290071457621,	4: 1.1214196657637756,	5: 1.0359980097164712,	6: 1.1171663256982962,	7: 0.8895313365757467,	8: 1.115403985157609,	9: 0.8699673360089462
                     
                     }  

GTG_Shapley_Value = {0: 2.7098433420062067, 1: 2.5451265975832933, 2: 2.468536647657553, 3: 2.6675533001621563, 4: 2.7696566904584574, 5: 2.196543390055498, 6: 2.195170027514299, 7: 2.8882866566379857, 8: 2.0823633059859277, 9: 2.392480008304119}

# 计算余弦相似度
# similarity = cosine_similarity(Exact_Shapley_Value, GTG_Shapley_Value) # 0.9997343798372547
# similarity = cosine_similarity(Exact_Shapley_Value, MAB_Shapley_Value) # 0.9867514726642874   
similarity = cosine_similarity(Exact_Shapley_Value, MAB_Shapley_Value) # 
print(f"cos_distance: {similarity}")

# max_diff = max_difference(Exact_Shapley_Value, GTG_Shapley_Value) # 0.07459421109219178
max_diff = max_difference(Exact_Shapley_Value, MAB_Shapley_Value) # 1.6448773020848861
# max_diff = max_difference(Exact_Shapley_Value, Delta_Shapley_Value) # 
print(f"max_distance: {max_diff}")