from google import genai

client = genai.Client(api_key="<api key goes here>")

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="""Generate a small python function using these input and output examples (only give the code, no explanation, no markdown style, no comments): 
    Input: [0,0] Output: 0,
    Input: [0,1] Output: 1,
    Input: [1,0] Output: 1,
    Input: [1,1] Output: 0"""
)
print(response.text)