def refine_sentence(raw_sentence: str) -> str:

    raw_sentence = (raw_sentence or "").strip()

    if not raw_sentence:
        return ""

    # ---------- Local Refinements (No Gemini Needed) ----------
    local_map = {
        "SORRY": "I'm sorry.",
        "THANK YOU": "Thank you.",
        "THANK_YOU": "Thank you.",
        "HELLO": "Hello!",
        "HI": "Hi!",
        "GOOD MORNING": "Good morning!",
        "GOOD_MORNING": "Good morning!",
        "GOOD AFTERNOON": "Good afternoon!",
        "GOOD_AFTERNOON": "Good afternoon!",
        "GOOD NIGHT": "Good night!",
        "GOOD_NIGHT": "Good night!",
        "WELCOME": "You're welcome.",
        "BYE": "Goodbye!",
        "PLEASE": "Please.",
        "YES": "Yes.",
        "NO": "No.",
        "HELP": "Please help me.",
        "STOP": "Please stop.",
        "COME": "Please come here.",
        "GO": "Please go ahead.",
        "WAIT": "Please wait.",
        "READ": "I want to read.",
        "WRITE": "I want to write.",
        "WORK": "I am working.",
        "EAT": "Let's eat.",
        "DRINK": "I would like a drink.",
        "WATER": "I need some water.",
        "I LOVE YOU": "I love you.",
        "ILOVEYOU": "I love you."
    }

    key = raw_sentence.upper()

    if key in local_map:
        return local_map[key]

    # ---------- Gemini ----------
    try:

        client = _get_client()

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=raw_sentence,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=120,
            ),
        )

        refined = (response.text or "").strip()

        refined = refined.strip('"').strip("'")

        if refined:
            return refined

    except Exception as e:

        print("⚠️ Gemini unavailable:", e)

    return _fallback(raw_sentence)