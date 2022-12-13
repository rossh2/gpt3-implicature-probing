# Scalar implicature probing for GPT-3

This work developed out of a final project for MIT 9.19 Computational Psycholinguistics with Curtis Chen, Sun Kim and Vighnesh Subramaniam, for whose thoughts, ideas and support I am deeply grateful.
In the final project, we focused on GPT-2 and BERT; this repository extends the scalar implicature part of the work to GPT-3.

I investigate whether GPT-3 is capable of calculating scalar implicatures using _some/all_ and exact numbers using both prompting techniques and surprisal measures. 
Results show that GPT-3 reliably calculates the scalar implicature for \emph{some/all} in a few-shot setting, as well as the number implicature, contrasting with GPT-2 
which was only able to calculate the scalar implicature for _some/all_ and failed on the number implicatures.

## What are scalar implicatures?

Conversational implicature is a phenomenon whereby a speaker says one thing and means another; 
typically, something else in addition (Grice, 1975; Simons, 2012). 
A classic example is scalar implicature, where speakers use items on a scale such as _none, some, all_
to convey that they mean exactly that much and no more. 
If a speaker says _Sam ate some of the cookies_, they typically mean that _Sam ate some but not all of the cookies_ -- 
a stronger statement using a pragmatically enriched meaning of _some_ 
The same phenomenon occurs with numbers: _Jo bought three of the toys_ means in most contexts that 
_Jo bought exactly three of the toys_ -- not four, five, or all of them. 
These are implicatures, not entailments, because they can be cancelled: a speaker can say 
_Sam ate some of the cookies, in fact, they were so good that he ate them all_. 
Typically, this phenomenon is explained in terms of alternatives: if the speaker meant that Sam ate all the cookies, 
then they could have said so (using _all_) and been more informative or accurate. 
Using _some_ when they could have used the stronger alternative _all_ violates Grice's Maxim of Quantity 
("Make your contribution as informative as is require"; Grice, 1975); 
thus, the listener concludes that when a speaker uses _some_, it must be because they can't use _all_, i.e. the speaker means _some and not all_.  

Implicatures form a classic example of "between the lines" meaning, a pragmatic inference which is understood by humans but not made explicit in the text.

## Implicature Processing in Humans

That adult humans calculate scalar implicatures is uncontroversial. 
The debate centres instead around two areas: first, how adults calculate the implicature 
(and how this interfaces with available theories of implicature calculation -- see Alatawi (2019) for a survey), and second, at what point children acquire this ability.

Here, we focus on a simple question-answer paradigm designed for testing children's acquisition of scalar implicatures developed by 
Sullivan et al. (2019). Sullivan et al. ran the experiment both on children and on adults, allowing us to compare our neural models to both. 
Sullivan et al. found that while children aged 4-7 may correctly answer that _some_ means _not all_, given the right context, they also infer that _all_ means 
_not some, unlike adults. Sullivan et al. argue that children are not actually calculating the scalar implicature, 
but just treating the two words _some_ and _all_ as mutually exclusive lexical items, in line with the more general Mutual Exclusivity Assumption for child word learning
(Au, 1987). 
We thus investigate whether our language models behave like human adults, human children, or fail to distinguish between _some_ and _all_ in either direction.

Specifically, Sullivan et al. presented adults and children with a puppet, Puppy, who had to complete a task and would get a prize if he completed the task. 
The task was described in words ("paint _all/some/two/three_ of the stars") in conjunction with a picture (of three unpainted stars), 
then Puppy's actions were described in words only ("Puppy painted _some_ of the stars"). 
Participants then chose whether to give Puppy a prize. 
We  adapt this task to neural language models.

## Prompt design

In the original squib and in this repository, we prompt the model(s) to answer questions about scalar implicatures, entailments, and matching controls in a range of settings, 
varying across the number of examples given before the target question (length of the prompt "prefix", between 0 and 6 examples), 
the exact prompt template, and type of scalar implicature (_some/all_ vs. number). 
In each case, the examples and target questions are randomly generated from a template-style "grammar". 
We average results across 50 trials for each configuration. 
Giving no examples at all before the target is a case of testing _zero-shot_ learning; a small quantity of examples is referred to as _few-shot_ learning. 

We test four types of prompts: direct questions, a Cloze template proposed by Schick and Schütze (2021), 
the prompting template used for textual entailment by GPT-3 (Brown et al. 2020), and the prize-giving scenario from Sullivan et al. (2019):

* **Direct questions:** Sam ate some of the cookies. Did Sam eat all of the cookies? Yes/No/Maybe.
* **Cloze task:** Sam ate some of the cookies? ___, Sam ate all of the cookies. Yes/No/Maybe.
* **GPT-3:** Sam ate some of the cookies. Question: Sam ate all of the cookies. True, False or Neither? True/False/Neither.
* **Prize:** If Sam eats some of the cookies, Sam will get a prize. Sam ate all of the cookies. Does Sam get a prize? Yes/No/Maybe.

Examples in the prompt prefix used the same template as the target, but contained the correct answer at the end. 
We used a template-based "grammar" to generate the examples and target questions with a range of verbs and verb-appropriate nouns. 
To focus the model's attention on entailments involving determiners, we used pairs of determiners that entailed or did not entail each other without the use of implicature, such as _all_ entailing _not none_, _two_ entailing _some_, and \emph{half} neither entailing nor contradicting _three_ (for a 'Maybe' answer). We balanced the division of the $n$ examples evenly between 'Yes' and 'No' answers with 1-2 'Maybe' answers (for $n$ odd vs. even). For $n=1$ we used a single 'Yes' example and for $n=2$ we used one 'Yes' and one 'No' example. This gave rise to the following templates:

* {det1} of the {noun} are {adj}. Are {det2} of the {noun} {adj}? {answer}.
* {subject} {past verb} {det1} of the {noun}. Did {subject} {inf verb} {det2} of the {noun}? {answer}.

Given the prefix of $n$ examples followed by the target prompt, we asked the model to deterministically predict an additional 2-5 tokens, and judged it correct if the tokens contained the target answer.

## Results on GPT-3

_Write-up in progress - short summary: GPT-3 does well at all kinds of implicatures, unlike GPT-2!_
