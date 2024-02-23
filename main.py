import sys
from helpers import write_json_to_file, prompt_openai, wrap_user_input, is_valid_json, Message


def main():
    

    initial_prompt = "testing the connection to OpenAI."
    print("Testing the connection to OpenAI...\n")

    try:
        res = prompt_openai([Message(role="system", content=initial_prompt)])
        if not res:
            print("Error connecting to OpenAI")
            sys.exit(1)
    except Exception as e:
        print("Error connecting to OpenAI:", e)
        return

    print("Welcome to the GPT Data Generator!")
    print("<--------------------------------->")
    print(
        "This program will help you generate JSON data to train a LoRA model using OpenAI's GPT-3.\n"
    )

    user_input = print(
        "What type of data would you like to generate?\nSome fun examples are: \n- A list of quotes from your favorite anime character\n- Dad jokes\n- Country song lyrics"
    )

    user_input = input("I want a dataset about: ")

    response_text = prompt_openai(wrap_user_input(user_input))

    if not is_valid_json(response_text):
        print("Sorry :( OpenAI response is not valid JSON. Please try again.")
        sys.exit(1)

    write_json_to_file(response_text)

    print(
        "Checkout the file called data.json. You took the first step to fine tuning a custom model ;)"
    )


if __name__ == "__main__":
    main()
