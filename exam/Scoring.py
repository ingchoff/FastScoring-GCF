def main_process(amount_answer, point, type_scoring, amount_correct):
    score = 0
    if amount_answer == 1:
        score += 1*point
    elif amount_answer > 1 and type_scoring == 'all':
        score += 1*point
    elif amount_answer > 1 and type_scoring == 'average':
        weight_per_choices = (100/amount_answer)
        score += weight_per_choices*point*amount_correct
    return score
