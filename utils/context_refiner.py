def refine_text(text):

    text = text.upper().strip()

    rules = {

        "HELLO": "Hello!",

        "THANK_YOU": "Thank you.",

        "I_LOVE_YOU": "I love you.",

        "HELP": "Please help me.",

        "PLEASE": "Please.",

        "SORRY": "I am sorry.",

        "GOOD": "Good.",

        "BAD": "That is bad.",

        "YES": "Yes.",

        "NO": "No."
    }

    # Exact match
    if text in rules:
        return rules[text]

    # Multi-word refinement
    words = text.split()

    refined = []

    for word in words:

        if word in rules:
            refined.append(rules[word])

        else:
            refined.append(word.capitalize())

    return " ".join(refined)