"""
Contain prompts to compare winrate
"""

SUMMARIZATION_WINRATE_S = """Which of the following summaries does a better job of summarizing the most
important points in the given forum post?

Post:
<post>

Summary A:
<Summary A>

Summary B:
<Summary B>

FIRST provide a one-sentence comparison of the two summaries, explaining which
you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your
choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">
"""

SUMMARIZATION_WINRATE_C = """Which of the following summaries does a better job of summarizing the most 
important points in the given forum post, without including unimportant or 
irrelevant details? A good summary is both precise and concise.

Post:
<post>

Summary A:
<Summary A>

Summary B:
<Summary B>

FIRST provide a one-sentence comparison of the two summaries, explaining which
you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your 
choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">
"""

DIALOGUE_WINRATE = """For the following query to a chatbot, which response is more helpful?

Query: {the_user_query}

Response A:
{either_the_test_method_or_baseline}

Response B:
{the_other_response}

FIRST provide a one-sentence comparison of the two responses and explain
which you feel is more helpful. SECOND, on a new line, state only "A" or
"B" to indicate which response is more helpful. Your response should use
the format:
Comparison: <one-sentence comparison and explanation>
More helpful: <"A" or "B">
"""