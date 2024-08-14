import os
import sk
from openai import OpenAI

client = OpenAI(api_key=sk.APIKEY)

# Upload training data to OpenAI
training_file = client.files.create(
  file = open("fine-tune/training-data_1.jsonl", "rb"),
  purpose = "fine-tune"
)

validation_file = client.files.create(
  file = open("fine-tune/validation-data_1.jsonl", "rb"),
  purpose = "fine-tune"
)

# Create a fine-tune model
client.fine_tuning.jobs.create(
    training_file = training_file.id,
    validation_file = validation_file.id,
    suffix = "Final-Year-Proj",
    model = "gpt-3.5-turbo"
)

# prompt = """You are a helpful assistant providing detailed information about the Data Mining module to university students. 
#     Limit your responses to the scope of the Data Mining, and if the question is not related to Data Mining, 
#     explicitly state that "This question is out of scope of this module" without attempting to provide an answer. 
#     If questions are being asked, provide a concise and accurate explanations as well."""

# response = client.chat.completions.create(
#     model="ft:gpt-3.5-turbo-0125:personal:final-year-proj:953J3u0R",
#     messages=[
#     {"role": "system", "content": prompt},
#     {"role": "user", "content": "How will the module be assessed"}
#     ]
# )

# print(dict(response)['choices'][0].message.content)