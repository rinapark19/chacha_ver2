import numpy as np

def calculate_normalized_score(g_eval_scores, token_probabilities):
    g_eval_scores = np.array(g_eval_scores)
    token_probabilities = np.array(token_probabilities)

    normalized_scores = g_eval_scores * token_probabilities

    normalized_scores = np.clip(normalized_scores, 1.0, 10.0)

    return normalized_scores

g_eval_scores = [[8, 9, 8]]
token_p = [0.015706806282722512, 0.005235602094240838, 0.015706806282722512]
normalized_score = calculate_normalized_score(g_eval_scores, token_p)