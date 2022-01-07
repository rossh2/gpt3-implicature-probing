from eval import eval_dets
from promptgen import SCALAR_DET_PAIRS, NUMBER_DET_PAIRS_TWO, \
    NUMBER_DET_PAIRS_FOUR, NUMBER_DET_PAIRS_THIRTEEN


def get_name(det_pairs):
    if det_pairs == SCALAR_DET_PAIRS:
        name = 'scalar-dets'
    elif det_pairs == NUMBER_DET_PAIRS_TWO:
        name = 'number-dets-two'
    elif det_pairs == NUMBER_DET_PAIRS_FOUR:
        name = 'number-dets-four'
    elif det_pairs == NUMBER_DET_PAIRS_THIRTEEN:
        name = 'number-dets-thirteen'
    else:
        raise ValueError('Set a name for these determiners')

    return name


if __name__ == '__main__':
    det_pairs = SCALAR_DET_PAIRS
    name = get_name(det_pairs)

    eval_dets(det_pairs, name, trials=25, min_prefix_length=0,
              max_prefix_length=4, engine='davinci', template_type='prize')

    # eval_dets(det_pairs, name, trials=1, min_prefix_length=1,
    #           max_prefix_length=1, engine='ada', template_type='prize')

    # eval_dets(SCALAR_DET_PAIRS, 'scalar-dets')
    # eval_dets(NUMBER_DET_PAIRS, 'number-dets',
    #           dets_to_plot=['two-three', 'three-at least two', 'three-three'])
    # eval_dets(SCALAR_DET_PAIRS, 'scalar-dets_ss', template_type='ss')
    # eval_dets(SCALAR_DET_PAIRS, 'scalar-dets_gpt3', template_type='gpt3')
    # eval_dets(SCALAR_DET_PAIRS, 'scalar-dets_prize', template_type='prize')
