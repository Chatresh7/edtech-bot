import google.generativeai as genai

genai.configure(api_key="AIzaSyD1jIsnO1SvEuNOvfLUnTIb-HPi2sxz9ys")

print("Available models for your API key:")
print("=" * 40)
for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(model.name)
