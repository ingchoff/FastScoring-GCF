def main_process(list_correct, point, type_scoring, list_pos):
    score = 0
    amount_correct = 0
    if len(list_pos) == 1:
        score = point
    elif len(list_pos) > 1 and type_scoring == 'all':
        score = point
    elif len(list_pos) > 1 and type_scoring == 'average':
        for j, pos in enumerate(list_correct):
            if len(list_pos) != 0 and pos in list_pos:
                amount_correct += 1
        weight_per_choices = (point/len(list_correct))
        score = weight_per_choices*amount_correct
    elif len(list_pos) > 1 and type_scoring == 'minimum':
        for j, pos in enumerate(list_correct):
            if len(list_pos) != 0 and pos in list_pos:
                score = point
    return score


def edit_score(data_result, type_scoring, point):
    final_score = 0
    print(data_result)
    for no, data in data_result.items():
        score = 0
        correct_clause = 0
        if data['correct_choice'] == data['user_choice'] and len(data['correct_choice']) == 1:
            score = int(point)
            data_result[no]['correct'] = int(point)
        elif data['correct_choice'] != data['user_choice'] and len(data['correct_choice']) == 1:
            score = 0
            data_result[no]['correct'] = 0
        elif data['correct_choice'] != data['user_choice'] and len(data['correct_choice']) > 1 and \
                type_scoring == 'all':
            score = 0
            data_result[no]['correct'] = 0
        elif data['correct_choice'] == data['user_choice'] and len(data['correct_choice']) > 1 and \
                type_scoring == 'all':
            score = int(point)
            data_result[no]['correct'] = int(point)
        elif data['correct_choice'] == data['user_choice'] and len(data['correct_choice']) > 1 and \
                type_scoring == 'average':
            score = int(point)
            data_result[no]['correct'] = int(point)
        elif data['correct_choice'] != data['user_choice'] and len(data['correct_choice']) > 1 and \
                type_scoring == 'average' and len(data['user_choice']) <= len(data['correct_choice']):
            for ans in data['correct_choice']:
                if ans in data['user_choice']:
                    correct_clause += 1
            weight_point = int(point)/len(data['correct_choice'])
            score = weight_point*correct_clause
            data_result[no]['correct'] = score
        elif  data['correct_choice'] == data['user_choice'] and len(data['correct_choice']) > 1 and \
                type_scoring == 'minimum':
            score = int(point)
            data_result[no]['correct'] = score
        elif data['correct_choice'] != data['user_choice'] and len(data['correct_choice']) > 1 and \
                type_scoring == 'minimum' and len(data['user_choice']) <= len(data['correct_choice']):
            for ans in data['correct_choice']:
                if ans in data['user_choice']:
                    score = int(point)
            score = score
            data_result[no]['correct'] = score
        final_score += score
        print(str(no) + ':' + str(score))
    print(final_score)
    print(data_result)
    return {
        'final_score': final_score,
        'new_data_result': data_result
    }
