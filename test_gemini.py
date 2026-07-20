from backend.gemini_service import refine_sentence

sentence = "TODAY ME GO WORK"

result = refine_sentence(sentence)

print("\nOriginal : ", sentence)
print("Gemini  : ", result)