from typing import List, TypedDict
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import sys

class Message(TypedDict):
    role: str
    content: str


def get_openai() -> OpenAI:
    load_dotenv()
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not openai_key:
        print(
            "OpenAI API key not found. Please set it in the OPENAI_API_KEY in the .env file."
        )
        sys.exit(1)
    return OpenAI(api_key=openai_key)


client = get_openai()


def write_json_to_file(json_data: str) -> None:
    with open("data.json", "w") as file:
        file.write(json_data)


def prompt_json_keys() -> List[str]:
    keys_input = input("Enter a list of JSON keys in the format [key_1,key_2]: ")
    keys_list = keys_input.strip("[]").split(",")
    keys_list = [key.strip() for key in keys_list]
    return keys_list


def is_valid_json(json_str: str) -> bool:
    try:
        json.loads(json_str)
    except ValueError as e:
        return False
    return True


def prompt_openai(messages: List[Message]) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # You can use other models as per your requirements
        temperature=0.7,
        n=1,
        stop=None,
        messages=messages,
    )

    return response.choices[0].message.content


def wrap_user_input(user_input: str) -> List[Message]:

    wrapped_prompt = f"""Your role is to generate json data based on the user's prompt there are no constraints on how many roles you should do so ensure to do as many as possible, if the data is based on real world events try to strive for accuracy, however if it's something humorous, whimsical or fantastical, take creative freedom where neccessary.
    
    Your response should be json and only json so that it can be easily parsed by a program. The data you respond with will be used to train a Low-Rank Adaptation LLM so format it accordingly.

    Here is the user's prompt: "{user_input}"
    """

    messages: List[Message] = [{"role": "user", "content": wrapped_prompt}]

    return messages
