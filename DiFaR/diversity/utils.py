from openai import OpenAI
import base64
import json
import os
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential

api_key = ""
base_url = ""  # fill your own api key and base url

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def construct_length(text):
    words = text.split()
    if len(words) <= 512:
        return text
    words = words[:512]
    text = ' '.join(words)
    return text


def error_process(state):
    return 'cannot obtain results'


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3), retry_error_callback=error_process)
def obtain_response(conversation):
    chat_completion = client.chat.completions.create(messages=conversation, model='gpt-4o-2024-08-06',
                                                     temperature=0.0
                                                     )
    output = chat_completion.choices[0].message.content
    return output
