import math
from random import randrange, shuffle
from typing import Tuple, List, Dict, TypeVar

T = TypeVar('T')
S = TypeVar('S')

EDIBLE = ['cupcakes', 'apples', 'sandwiches', 'cakes', 'oranges',
          'pancakes', 'pretzels', 'strawberries', 'chocolates', 'churros',
          'bagels', 'pastries', 'burgers', 'peanuts', 'cherries', 'snacks']
COLOURFUL = ['sweaters', 'candles', 'backpacks', 'scarves',
             'cars', 'shirts', 'bottles', 'shoes', 'tickets', 'hats',
             'umbrellas', 'mugs', 'keyrings', 'toys', 'rings']
BUYABLE = EDIBLE + COLOURFUL
VISIBLE = BUYABLE + ['mice', 'cats', 'giraffes', 'parades', 'stores',
                     'performers', 'dogs', 'sights', 'attractions',
                     'temples', 'adverts', 'warnings', 'signs']
OWNABLE = BUYABLE

VERB_OBJECT = {
    'ate': EDIBLE,
    'bought': BUYABLE,
    'sold': BUYABLE,
    'purchased': BUYABLE,
    'got': BUYABLE,
    'ordered': BUYABLE,
    'saw': VISIBLE,
    'noticed': VISIBLE,
    'spotted': VISIBLE,
    'had': OWNABLE,
    'owned': OWNABLE,
    'wanted': OWNABLE,
    'liked': OWNABLE,
}

VERB_PAST_INF = {
    'ate': 'eat',
    'bought': 'buy',
    'sold': 'sell',
    'purchased': 'purchase',
    'got': 'get',
    'ordered': 'order',
    'saw': 'see',
    'noticed': 'notice',
    'spotted': 'spot',
    'had': 'have',
    'owned': 'own',
    'wanted': 'want',
    'liked': 'like'
}

VERB_PAST_3SG = {
    'ate': 'eats',
    'bought': 'buys',
    'sold': 'sells',
    'purchased': 'purchases',
    'got': 'gets',
    'ordered': 'orders',
    'saw': 'sees',
    'noticed': 'notices',
    'spotted': 'spots',
    'had': 'has',
    'owned': 'owns',
    'wanted': 'wants',
    'liked': 'likes'
}

BE_PRES = 'are'

ADJECTIVE_NOUN = {
    'tasty': EDIBLE,
    'mouldy': EDIBLE,
    'cold': EDIBLE,
    'warm': EDIBLE,
    'delicious': EDIBLE,
    'expensive': BUYABLE,
    'cheap': BUYABLE,
    'affordable': BUYABLE,
    'sold out': BUYABLE,
    'on sale': BUYABLE,
    'impressive': VISIBLE,
    'pretty': VISIBLE,
    'colourful': VISIBLE,
    'green': COLOURFUL,
    'red': COLOURFUL,
    'orange': COLOURFUL,
    'blue': COLOURFUL,
    'yellow': COLOURFUL,
}

SUBJECTS = ['Sally', 'Ann', 'Laura', 'Mary', 'Lucy', 'Sophie',
            'Mark', 'Jim', 'John', 'Adam', 'Michael', 'Bill',
            'Sam', 'Jo', 'Sidney', 'Robin',
            'the teacher', 'the actor', 'the actress', 'the man', 'the woman',
            'the singer']

DETERMINER_PAIRS = {
    # Sort these by true/false first to make it easy to draw balanced samples
    'Yes': {
        'two': [
            'two',
            'some',
            'more than one',
        ],
        'both': [
            'two',
            'all',
            'both',
        ],
        'half': [
            'some',
            'half',
        ],
        'some': [
            'two',
            'more than one',
        ],
        'three': [
            'more than two',
            'three',
        ],
    },
    'No': {
        'none': [
            'two',
            'several',
            'some',
            'all'
        ],
        'exactly two': [
            'none',
            'three',
            'one',
        ],
        'several': [
            'none',
        ],
        'most': [
            'none',
        ],
        'all': [
            'none'
        ],
    },
    'Maybe': {
        'two': [
            'all',
            'half'
        ],
        'half': [
            'three',
        ],
        'most': [
            'three',
        ]
    }
}

for verb in VERB_OBJECT:
    assert (verb in VERB_PAST_INF), f'{verb} does not have an infinitive form'

verb_list = list(VERB_OBJECT.keys())
verb_count = len(verb_list)

adj_count = len(ADJECTIVE_NOUN)
subj_count = len(SUBJECTS)

SCALAR_DET_PAIRS = [
    ['some', 'all', 'No'],
    ['all', 'some', 'Yes'],
    ['all', 'all', 'Yes'],
    ['some', 'some', 'Yes']
]

NUMBER_DET_PAIRS_TWO = [
    # Implicatures and entailments
    ['two', 'three', 'No'],
    ['three', 'two', 'Yes'],  # Contentious if we read two as 'exactly two'
    ['three', 'at least two', 'Yes'],
    # Maybes
    ['at least two', 'three', 'Maybe'],
    # Sanity checks
    ['three', 'three', 'Yes'],
    ['at least two', 'at least two', 'Yes'],
]

NUMBER_DET_PAIRS_FOUR = [
    # Implicatures and entailments
    ['four', 'five', 'No'],
    ['five', 'four', 'Yes'],  # Contentious
    ['five', 'at least four', 'Yes'],
    # Maybes
    ['at least four', 'five', 'Maybe'],
    # Sanity checks
    ['five', 'five', 'Yes'],
    ['at least four', 'at least four', 'Yes']
]

NUMBER_DET_PAIRS_THIRTEEN = [
    # Implicatures and entailments
    ['thirteen', 'sixteen', 'No'],
    ['sixteen', 'thirteen', 'Yes'],  # Contentious
    ['sixteen', 'at least thirteen', 'Yes'],
    # Maybes
    ['at least thirteen', 'sixteen', 'Maybe'],
    # Sanity checks
    ['sixteen', 'sixteen', 'Yes'],
    ['at least thirteen', 'at least thirteen', 'Yes']
]

be_bias = 3  # Count 'be' as this many verbs so that we get more
# 'The X are Y' constructions relative to 'X Ved Y'

gpt3_ans_map = {
    'Yes': 'True',
    'No': 'False',
    'Maybe': 'Neither',
}

opposing_answer_map = {
    'Yes': 'No',
    'No': 'Yes',
    'True': 'False',
    'False': 'True',
    'Maybe': 'Yes',  # Define for completeness
    'Neither': 'True',  # Define for completeness
}


def get_answers(template_type: str) -> List[str]:
    if template_type == 'gpt3':
        return list(gpt3_ans_map.values())
    else:
        return list(gpt3_ans_map.keys())


def draw_from_dict(dic: Dict[T, S]) -> Tuple[T,S]:
    # Rely on dictionaries being sorted in Python 3.6+
    key_list = list(dic.keys())
    idx = randrange(0, len(key_list))
    key = key_list[idx]
    return key, dic[key]


def construct_question_sentence(first_det: str, second_det: str, answer: str,
                                template_type='qa') -> str:
    """
    Template types:
    qa: Tom ate some of the cookies. Did Tom eat all of the cookies?
    ss: Tom ate all of the cookies? __, Tom ate some of the cookies.
        (Schick & Schuetze, 2021)
    gpt3: Tom ate some of the cookies. Question: Tom ate all of the cookies.
          True, False, or Neither?
    prize: If Tom eats all of the cookies, Tom will get a prize.
           Tom ate some of the cookies. Does Tom get a prize?
    """
    if template_type not in ['qa', 'ss', 'gpt3', 'prize']:
        raise ValueError('Template type not supported')

    verb_idx = randrange(0, verb_count + be_bias)

    if verb_idx >= verb_count:
        # Use main verb be + adjective
        adj, noun_list = draw_from_dict(ADJECTIVE_NOUN)
        noun_idx = randrange(0, len(noun_list))
        noun = noun_list[noun_idx]

        if template_type == 'qa':
            sent = f'{first_det} of the {noun} are {adj}. ' \
                   f'Are {second_det} of the {noun} {adj}? {answer}.'
        elif template_type == 'ss':
            sent = f'{first_det} of the {noun} are {adj}? ' \
                   f'__, {second_det} of the {noun} are {adj}. {answer}.'
        elif template_type == 'gpt3':
            second_det_cap = second_det[0].upper() + second_det[1:]
            gpt_ans = gpt3_ans_map[answer]
            sent = f'{first_det} of the {noun} are {adj}. ' \
                   f'Question: {second_det_cap} of the {noun} are {adj}. ' \
                   f'True, False or Neither? {gpt_ans}.'
        elif template_type == 'prize':
            first_det_cap = first_det[0].upper() + first_det[1:]
            sent = f'If {second_det} of the {noun} are {adj}, ' \
                   f'you will get a prize. ' \
                   f'{first_det_cap} of the {noun} are {adj}. ' \
                   f'Do you get a prize? {answer}.'
        # noinspection PyUnboundLocalVariable
        sent = sent[0].upper() + sent[1:]  # Capitalise first letter
        return sent

    else:
        verb = verb_list[verb_idx]

        noun_list = VERB_OBJECT[verb]
        noun_idx = randrange(0, len(noun_list))
        noun = noun_list[noun_idx]

        subj_idx = randrange(0, subj_count)
        subject = SUBJECTS[subj_idx]

        past_verb = verb
        inf_verb = VERB_PAST_INF[verb]
        sg3_verb = VERB_PAST_3SG[verb]

        if template_type == 'qa':
            sent = f'{subject} {past_verb} {first_det} of the {noun}. ' \
                   f'Did {subject} {inf_verb} {second_det} of the {noun}? ' \
                   f'{answer}.'
        elif template_type == 'ss':
            sent = f'{subject} {past_verb} {first_det} of the {noun}? ' \
                   f'__, {subject} {past_verb} {second_det} of the {noun}. ' \
                   f'{answer}.'
        elif template_type == 'gpt3':
            subject_cap = subject[0].upper() + subject[1:]
            gpt_ans = gpt3_ans_map[answer]
            sent = f'{subject} {past_verb} {first_det} of the {noun}. ' \
                   f'Question: {subject_cap} {past_verb} {second_det} ' \
                   f'of the {noun}. True, False or Neither? {gpt_ans}.'
        elif template_type == 'prize':
            subject_cap = subject[0].upper() + subject[1:]
            sent = f'If {subject} {sg3_verb} {second_det} of the {noun}, ' \
                   f'{subject} will get a prize. ' \
                   f'{subject_cap} {past_verb} {first_det} of the {noun}. ' \
                   f'Does {subject} get a prize? {answer}.'
        # noinspection PyUnboundLocalVariable
        sent = sent[0].upper() + sent[1:]  # Capitalise first letter
        return sent


def construct_question_prompt(answer: str, template_type='qa') -> str:
    first_det, det_list = draw_from_dict(DETERMINER_PAIRS[answer])
    second_det_idx = randrange(0, len(det_list))
    second_det = det_list[second_det_idx]

    return construct_question_sentence(first_det, second_det, answer,
                                       template_type=template_type)


def generate_question_prompts(n=5, balanced=True, template_type='qa') -> str:
    prompt_sents = []

    if n == 0:
        return ''
    elif n == 1:
        yes_count = 1
        no_count = 0
    elif n == 2:
        yes_count = 1
        no_count = 1
    elif balanced:
        yes_count = math.floor((n - 1) / 2)
        no_count = yes_count
    else:
        yes_count = randrange(0, n)
        no_count = (n - 1) - yes_count

    # Sample yes
    for i in range(yes_count):
        prompt_sents.append(
            construct_question_prompt('Yes', template_type=template_type))

    # Sample no
    for i in range(no_count):
        prompt_sents.append(
            construct_question_prompt('No', template_type=template_type))

    # Add one or two examples of Maybe
    for i in range(n - len(prompt_sents)):
        prompt_sents.append(
            construct_question_prompt('Maybe', template_type=template_type))

    shuffle(prompt_sents)
    return '\n'.join(prompt_sents) + '\n'


def split_question_answer(sentence: str, template_type='qa') \
        -> Tuple[str, str]:
    if template_type == 'qa' or template_type == 'gpt3' \
            or template_type == 'prize':
        split = sentence.split('?')
        return split[0] + '?', strip_answer(split[1])
    elif template_type == 'ss':
        split = sentence.split('.')
        if len(split) == 3:
            # answer contains a period
            return split[0] + '.', strip_answer(split[1])
        else:
            return split[0] + '.', strip_answer(split[1])
    else:
        raise ValueError('Unsupported template type')


def strip_answer(answer: str) -> str:
    return answer.strip().replace('.', '')
