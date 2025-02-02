import io
import sys
import json
import base64
import tempfile
import pdf2image
from typing import List
from PIL.Image import Image
from datetime import datetime
import pdf2image.exceptions
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument


# This will be created if an OpenAI API key is entered for ease of use.
TEMP_ENV_FILE = ".env_temp"

# Required prompt to help the model to create a json object.
USER_PROMPT = (
    "Considering the information provided to you, provide me the followings. "
    "\n\nHere are the required json keys for bank and customer details:\n"
    "- \"isBankStatement\" (boolean) determines whether it is in fact a bank "
    "statement, or some other kind of document like a water bill, "
    "electricity bill or gsm provider bill. If it is a bill or something "
    "similar, return `false`. If it is a bank statement, return `true`.\n"
    "- \"account_holder_name\" (string) determines the name of the business "
    "that the document is addressed to. This is not the name of the bank "
    "that the customer or the business is working with.\n"
    "- \"account_holder_address\" (string) determines the address of the "
    "customer or the business on the bank statement.\n"
    "- \"account_holder_net_balance\" (float) determines the net change "
    "in account balance according to the transactions and whether or not "
    "this reconciles with the balances present on the document.\n"
    "- \"bank_name\" (string) determines the bank that the bank statement "
    "belongs to. If you cannot determine it, fill with `null`.\n"
    "- \"bank_address\" (string) determines the bank that the statement "
    "belongs to. If you cannot determine it or the info does not exist "
    "on the statement, fill with `null`."
    "\n\nHere are the required json keys to determine possible fraud signals:\n"
    "- \"is_modified\" (boolean) determines whether the document is modified "
    "or falsified or is it in the original form as the bank provided to their "
    "customers.\n"
    "- \"has_overlay\" (boolean) determines he structure of the PDF includes "
    "text that is hidden or overlaid.\n"
    "- \"is_template\" (boolean) determines the document contains obvious "
    "placeholders from a template.\n\n"
    "Finally, think step by step to determine if this is a bank statement "
    "or not. If it is not, make sure \"isBankStatement\" is `false`.\n\n"
    "Here is an example json template:\n"
)

JSON_FORMAT = """
{
    "isBankStatement": false,
    "account_holder_name": null,
    "account_holder_address": null,
    "account_holder_net_balance": null,
    "bank_name": null,
    "bank_address": null,
    "is_modified": null,
    "has_overlay": null,
    "is_template": null
}
"""

METADATA = """
Here is the metadata to determine possible fraud signals:
{metadata}
"""

SYSTEM_PROMPT = (
    "You are a helpful assistant extracting text from pdf files provided to "
    "you. Your job is to read the information on the pdf file and answer the "
    "user questions according to that information. The output format should "
    "be in json format. "
    "Do not answer anything except this json format. "
    "Do not add pleasantries or greetings. "
    "Do not add any keys except the given format show below. "
)


def clean_output(dirty_json: str) -> dict:
    """
    Cleans the LLM output to form a proper json.
    
    Args:
        dirty_json (str): Text created by the LLM model.
    Returns:
        clean_json (dict): Cleaned output.
    """
    # Clean the markdown output
    clean_json = dirty_json\
        .replace("```json\n", '')\
        .replace("\n```", '')
    try:
        clean_json = json.loads(s=clean_json)
        if not clean_json["isBankStatement"]:
            return json.loads(s=JSON_FORMAT)
        return clean_json
    except json.JSONDecodeError as e:
        print("[SYSTEM INFO] The output is not a valid JSON. Please try again.")
        return {}


def parse_pdf(file_path: str) -> Image:
    """
    Gets the pdf file path and converts it to images.
    Also, extract its metadata.
    
    Args:
        pdf_file_path (str): Path to the pdf file.
    Returns:
        tuple(Image, dict): Tuple of image file transformed from pdf and its metadata.
    """
    # creates a temporary folder to keep the digested pdf file
    with tempfile.TemporaryDirectory() as temp_folder:
        imgs = pdf2image.convert_from_path(
            pdf_path=file_path,
            output_folder=temp_folder
        )
    return (imgs[0].convert("RGB"), extract_metadata(file_path))


def extract_metadata(file_path: str) -> dict:
    """
    Extract metadata from a pdf file.

    Args:
        file_path (str): Path to pdf file.

    Returns:
        metadata (dict): Metadata extracted from pdf file.
    """
    with open(file=file_path, mode="rb") as pdf:
        parser = PDFParser(fp=pdf)
        pdf_doc = PDFDocument(parser=parser)
        metadata = pdf_doc.info[0]
        try:
            # "CreationDate" and "ModDate" (Modified Date) are not in human-readable format.
            # Let's convert them for ease of use. e.g., b"D:20060308133638-06'00'" ---> "2006-03-08 13:36:38"
            creation_date = metadata["CreationDate"].decode("utf-8")[2:][:14]
            mod_date = metadata["ModDate"].decode("utf-8")[2:][:14]

            # convert to propert datetime format and overwrite the metadata values
            creation_date = datetime.strptime(creation_date, "%Y%m%d%H%M%S")
            mod_date = datetime.strptime(mod_date, "%Y%m%d%H%M%S")
            metadata["CreationDate"] = creation_date.strftime("%Y-%m-%d %H:%M:%S")
            metadata["ModDate"] = mod_date.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
           print(
               "[SYSTEM INFO] Dates in the metadata could not be converted to human "
               "readable format. Most possibly the model will understand it."
            )
        finally:
            return metadata


def qwen_chat_template(file_path: str) -> List[dict]:
    """
    Gets the pdf file path, transforms it and creates the message template.
    
    Args:
        file_path (str): Location of the pdf file
    Returns:
        List[dict]: Message template to be fed into Qwen model.
    """
    try:
        image, metadata = parse_pdf(file_path=file_path)
    except pdf2image.exceptions.PDFPageCountError as e:
        print("\n[SYSTEM INFO]", e)
        sys.exit()

    return [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image
            },
            {
                "type": "text",
                "text": (
                    USER_PROMPT + 
                    JSON_FORMAT + 
                    METADATA.format(metadata=metadata)
                )
            }
        ],
    }
]


def openai_chat_template(file_path: str) -> List[dict]:
    """
    Gets the pdf file path, transforms it and creates the message template.
    
    Args:
        file_path (str): Location of the pdf file
    Returns:
        List[dict]: Message template to be fed into ChatGPT model.
    """
    try:
        image, metadata = parse_pdf(file_path=file_path)
    except pdf2image.exceptions.PDFPageCountError as e:
        print("\n[SYSTEM INFO]", e)
        sys.exit()
    # create a temporary buffer to keep the transformed image
    buffer = io.BytesIO()
    image.save(fp=buffer, format="JPEG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        USER_PROMPT + 
                        JSON_FORMAT + 
                        METADATA.format(metadata=metadata)
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ],
        }
    ]
