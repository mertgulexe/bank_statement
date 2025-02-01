import os
import sys
import json
from time import sleep
from dotenv import load_dotenv
from ModelVault import QwenModel, OpenAIModel
from DataProcess import (
    qwen_chat_template,
    openai_chat_template,
    TEMP_ENV_FILE
)


is_env_loaded = load_dotenv(TEMP_ENV_FILE)

os.system("clear")
print("""
**************************************************************
    Hello.
    This is an AI model that parses a bank statement
    uploaded as a PDF file and provides information about it.
**************************************************************
"""
)
sleep(1)
print("""\
Let's start by locating the PDF file. Without using any quotation marks, \
please provide the file path. If the file is not in the same directory as \
the `app.py` file, an absolute path is required:\
"""
)
PATH_TO_PDF = input('(default is "test_statement.pdf"): ') or "test_statement.pdf"
print("""
Choose a model.
1) "Qwen2-VL-2B" (lower performance, downloads the model for the first time)
2) "gpt-4o-mini" (higher performance, requires API key but no download)\
"""
)
chosen_model = input("(default is 2): ") or '2'

if chosen_model == '1':
    messages = qwen_chat_template(file_path=PATH_TO_PDF)
    model = QwenModel()
elif chosen_model == '2':
    messages = openai_chat_template(file_path=PATH_TO_PDF)
    if is_env_loaded:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    else:
        OPENAI_API_KEY = input(
            "\nPlease enter the OPENAI_API_KEY "
            "(the key will be saved locally and the screen will be cleared "
            "for privacy reasons): "
        )
        os.system("clear")
        os.system(f'echo OPENAI_API_KEY=\"{OPENAI_API_KEY}\" > {TEMP_ENV_FILE}')
    model = OpenAIModel(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini"
    )
else:
    print("Please choose the model 1 or 2.")
    sys.exit()

# final output
json_output = model.generate(messages=messages)
# delete unnecessary keys
del json_output["bank_name"]
del json_output["bank_address"]

print("Output:")
print(json.dumps(obj=json_output, indent=4))
