def refine_text(words):

    if not words:
        return ""

    refined_sentences = []

    for word in words:

        if word == "HELLO":
            refined_sentences.append("Hello!")

        elif word == "HELP":
            refined_sentences.append("Please help me.")

        elif word == "THANK_YOU":
            refined_sentences.append("Thank you.")

        elif word == "I_LOVE_YOU":
            refined_sentences.append("I love you.")

        elif word == "SORRY":
            refined_sentences.append("I am sorry.")

        elif word == "YES":
            refined_sentences.append("Yes.")

        elif word == "NO":
            refined_sentences.append("No.")

        else:
            refined_sentences.append(word.capitalize())

    return " ".join(refined_sentences)