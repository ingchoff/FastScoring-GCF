def main_process(amount_answer, point, type_scoring, amount_correct):
    score = 0
    if amount_answer == 1:
        score += 1*point
    elif amount_answer > 1 and type_scoring == 'all':
        score += 1*point
    elif amount_answer > 1 and type_scoring == 'average':
        weight_per_choices = (point/amount_answer)
        score += weight_per_choices*amount_correct
    return score


def edit_score(data_result, type_scoring, point):
    final_score = 0
    print(data_result)
    for no, data in data_result.items():
        score = 0
        correct_clause = 0
        if data['correct'] and data['correct_choice'] == data['user_choice'] and len(data['correct_choice']) == 1:
            score = int(point)
        elif not data['correct'] and data['correct_choice'] != data['user_choice'] and len(data['correct_choice']) == 1:
            score = 0
        elif not data['correct'] and data['correct_choice'] != data['user_choice'] and len(data['correct_choice']) > 1 and \
                type_scoring == 'all':
            score = 0
        elif data['correct'] and data['correct_choice'] == data['user_choice'] and len(data['correct_choice']) > 1 and \
                type_scoring == 'all':
            score = int(point)
        elif data['correct'] and data['correct_choice'] == data['user_choice'] and len(data['correct_choice']) > 1 and \
                type_scoring == 'average':
            score = int(point)
        elif not data['correct'] and data['correct_choice'] != data['user_choice'] and len(data['correct_choice']) > 1 and \
                type_scoring == 'average' and len(data['user_choice']) <= len(data['correct_choice']):
            for ans in data['correct_choice']:
                if ans in data['user_choice']:
                    correct_clause += 1
            weight_point = int(point)/len(data['correct_choice'])
            score = weight_point*correct_clause
        final_score += score
        print(str(no) + ':' + str(score))
    print(final_score)
    return final_score
