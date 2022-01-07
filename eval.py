import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Set

import numpy as np
import openai
import pandas as pd
import seaborn as sns
from matplotlib.collections import PathCollection
from openai.openai_object import OpenAIObject

from promptgen import generate_question_prompts, construct_question_sentence, \
    split_question_answer, get_answers, opposing_answer_map, strip_answer

openai.api_key = os.getenv('OPENAI_API_KEY')


def generate_text_from_prompt(prompt: str, max_new_tokens: int,
                              template_type: str,
                              engine='ada') -> Tuple[
    str, Dict[str, Optional[float]]]:
    '''
    Engines: 'davinci' is the full GPT-3 model with 175B parameters
    'ada' has just 350M parameters (smaller than GPT-2 which has 1.5B)
    but is cheap for testing
    '''
    response = openai.Completion.create(
        engine=engine,  # use 'ada' while testing new functionality, '
        prompt=prompt,
        max_tokens=max_new_tokens,
        n=1,  # only generate one response
        temperature=0,  # argmax sampling
        logprobs=5,  # return logprobs of m most likely tokens, max is 5
    )
    completion = response.choices[0]  # Since n = 1
    text = completion.text

    first_word_logprobs = completion.logprobs.top_logprobs[0]
    answer_logprobs = clean_logprobs(first_word_logprobs, template_type)

    return text, answer_logprobs


def evaluate_model(det_pairs: List[List[str]],
                   min_prefix_length: int, max_prefix_length: int, trials=1,
                   print_outputs=True, fuzzy_match=True, template_type='qa',
                   engine='davinci') -> pd.DataFrame:
    possible_answers = get_answers(template_type)
    results_df = pd.DataFrame(
        columns=['PrefixLength', 'Determiners', 'Prompt', 'Answer',
                 'Correct'] + [f'{a}LogProb' for a in possible_answers]
    )

    # With short prompts, GPT will sometimes generate things like '\n\nA: Yes.'
    # instead of 'Yes.'
    # If fuzzy_match is true, give it a few extra tokens and check if the
    # correct answer is contained in the output.
    # Otherwise, allow exactly two tokens (the answer plus a period)
    max_new_tokens = 5 if fuzzy_match else 2

    for prefix_length in range(min_prefix_length, max_prefix_length + 1):
        print(f'Prefix length: {prefix_length}')

        for i in range(trials):
            # Hold prefix constant per trial
            prefix = generate_question_prompts(prefix_length,
                                               template_type=template_type)

            for first_det, second_det, answer in det_pairs:
                test_s = construct_question_sentence(
                    first_det, second_det,
                    answer,
                    template_type=template_type
                )
                prompt, answer = split_question_answer(test_s, template_type)

                gen_answer, logprobs = generate_text_from_prompt(
                    prefix + prompt,
                    max_new_tokens=max_new_tokens,
                    template_type=template_type,
                    engine=engine,
                )

                if print_outputs:
                    print_answer(prompt, gen_answer)

                this_result = {
                    'PrefixLength': [prefix_length],
                    'Determiners': [f'{first_det}-{second_det}'],
                    'Prompt': [prefix + prompt],
                    'Answer': gen_answer,
                    'Correct': int(answer in gen_answer),
                    'SurprisalDelta': calculate_surprisal_delta(answer,
                                                                logprobs)
                }
                for a in possible_answers:
                    this_result[f'{a}LogProb'] = logprobs[a]

                results_df = results_df.append(pd.DataFrame(this_result))

    results_df.reset_index(inplace=True, drop=True)
    return results_df


def clean_logprobs(logprobs: OpenAIObject, template_type: str) \
        -> Dict[str, Optional[float]]:
    possible_answers = get_answers(template_type)

    clean_logprobs = defaultdict(lambda: np.nan)
    for a in logprobs.keys():
        clean_a = strip_answer(a)
        if clean_a in possible_answers:
            clean_logprobs[clean_a] = logprobs[a]

    return clean_logprobs


def calculate_surprisal_delta(answer: str,
                              logprobs: Dict[str, Optional[float]]) \
        -> Optional[float]:
    wrong_answer = opposing_answer_map[answer]
    if answer in logprobs and wrong_answer in logprobs:
        # Surprisal = -logprob, so add minus in front of whole expression
        # Model should be more surprised for wrong answer,
        # so positive delta = correct
        delta = - (logprobs[wrong_answer] - logprobs[answer])
        return delta
    else:
        return np.nan


def get_accuracy_from_results(results_df: pd.DataFrame) -> pd.DataFrame:
    accuracy_df = pd.DataFrame(
        results_df.groupby(['PrefixLength', 'Determiners'])[
            'Correct'].mean())
    accuracy_df.reset_index(inplace=True)
    accuracy_df.rename(columns={'Correct': 'Accuracy'}, inplace=True)

    return accuracy_df


def print_answer(prompt: str, gen_answer: str) -> None:
    # Remove extra tokens from fuzzy matching if irrelevant
    print_answer = gen_answer
    for expected in ['Yes.', 'No.', 'Maybe.']:
        if gen_answer.strip().startswith(expected):
            print_answer = expected
            break

    print(prompt + ' ' + print_answer)


def make_filename(name: str, template_type: str, engine: str,
                  trials: int, min_prefix_length: int,
                  max_prefix_length: Optional[int] = None) -> str:
    timestamp = datetime.now().strftime('%Y-%m-%dT%H%M')
    if max_prefix_length:
        return f'results/{engine}_prefix{min_prefix_length}-{max_prefix_length}' \
               f'_{trials}trials_{name}_{template_type}_{timestamp}'
    else:
        return f'results/{engine}_prefix{min_prefix_length}' \
               f'_{trials}trials_{name}_{template_type}_{timestamp}'


def plot_accuracy(results: pd.DataFrame, trials: int,
                  min_prefix_length: int, max_prefix_length: int,
                  filename, dets_to_plot: List[str] = []):
    if dets_to_plot:
        data = results[results['Determiners'].isin(dets_to_plot)]
    else:
        data = results

    g = sns.relplot(x='PrefixLength', y='Accuracy', hue='Determiners',
                    kind='line',
                    markers=True,
                    data=data)
    g.set(title=f'Accuracy with {trials} trials per determiner pair',
          xticks=range(min_prefix_length, max_prefix_length + 1))

    g.savefig(f'{filename}.png', bbox_inches='tight')


def plot_surprisaldelta(results_df: pd.DataFrame, prefix_length,
                        trials, filename):
    if 'SurprisalDelta' not in results_df.columns:
        # Data created by older version of script
        if 'LogProbDelta' in results_df.columns:
            results_df['SurprisalDelta'] = -results_df['LogProbDelta']
        else:
            # Nothing to plot here, surprisal was not stored
            return

    filtered_data = results_df.loc[
        results_df['PrefixLength'] == prefix_length]

    g2 = sns.violinplot(x='Determiners', y='SurprisalDelta', color='0.75',
                        data=filtered_data)

    for artist in g2.lines:
        artist.set_zorder(10)
    for artist in g2.findobj(PathCollection):
        artist.set_zorder(11)

    sns.swarmplot(x='Determiners', y='SurprisalDelta',
                  data=filtered_data, ax=g2)

    g2.axhline(0, color='0.1')

    g2.set(
        title=f'Delta between false and correct answer surprisal\nwith {trials} '
              f'trials per determiner pair (prefix length {prefix_length})')

    g2.figure.savefig(f'{filename}_prefix{prefix_length}-surprisaldelta.png',
                      bbox_inches='tight')


def eval_dets(dets: List[List[str]], name: str, template_type='qa',
              trials=50, min_prefix_length=0, max_prefix_length=4,
              surprisal_prefix_lengths: Set[int] = None,
              engine='davinci',
              dets_to_plot: List[str] = []):
    if not surprisal_prefix_lengths:
        surprisal_prefix_lengths = {min_prefix_length, 3}

    filename = make_filename(name, template_type, engine, trials,
                             min_prefix_length, max_prefix_length)
    print(f'Engine: {engine}\n'
          f'Trials: {trials}\n'
          f'Prefixes {min_prefix_length}-{max_prefix_length}\n'
          f'Template type: {template_type}\n'
          f'Stored under {filename}.csv/png\n')

    results_df = evaluate_model(dets,
                                min_prefix_length=min_prefix_length,
                                max_prefix_length=max_prefix_length,
                                trials=trials,
                                fuzzy_match=False,
                                template_type=template_type,
                                engine=engine)
    results_df.to_csv(f'{filename}.csv', index=False)

    accuracy_df = get_accuracy_from_results(results_df)
    print(accuracy_df)
    accuracy_df.to_csv(f'{filename}_accuracy.csv', index=False)

    plot_accuracy(accuracy_df, trials, min_prefix_length,
                  max_prefix_length, filename, dets_to_plot=dets_to_plot)

    for surprisal_prefix_length in surprisal_prefix_lengths:
        if min_prefix_length <= surprisal_prefix_length <= max_prefix_length:
            plot_surprisaldelta(results_df, surprisal_prefix_length,
                                trials, filename)


if __name__ == '__main__':
    trials = 25
    filename = 'results/davinci_prefix0-4_25trials_scalar-dets_prize_2021-12-17T1349'
    results = pd.read_csv(f'{filename}.csv')
    plot_surprisaldelta(results, 1, trials, filename)
