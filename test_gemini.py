from google import genai

client = genai.Client(
    api_key="AIzaSyByNztOSuuYItTn6QxjnJkcnNvE0M7Wlo0"
)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Convert this into a meaningful sentence: HELLO THANK_YOU HELP"
)

print(response.text)