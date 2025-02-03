# Bank Statement AI Parser
## 1. Overview
This parser is an LLM-powered CLI designed to parse a bank statement PDF file and extract key information such as account details and transaction data, while also detecting any fraudulent activities. It also supports two different models:
* `Qwen2-VL-2B`: A free, open-source model that leads the VQA (Visual Question Answering) benchmarks.
* `gpt-4o-mini`: A paid but higher performance model that requires an API key.

The output format is a JSON object, and an example is shown below:
```json
{
    "isBankStatement": true,
    "account_holder_name": "Mr Sherlock Holmes",
    "account_holder_address": "221B Baker Street, London, NW1 6XE",
    "account_holder_net_balance": 13579.42,
    "is_modified": false,
    "has_overlay": true,
    "is_template": false
}
```

## 2. Design Environment
- **Python Versions Tested**: `3.12.3` and `3.13.1`
- **Operating System**: Ubuntu 24.04.1 LTS
- **GPU**: NVIDIA GeForce RTX 4070 [Laptop]
    - **Memory**: 8 GB
    - **Driver Version**: 535.183.01
    - **CUDA Version**: 12.2    
- **Date**: February 2, 2025

## 3. Installation
Before running the tool, you need to install the required dependencies. Open your terminal in the project directory and run:

```bash
./requirements_install.sh
```

This script will install all required Python dependencies to run the project. ***Using a virtual environment is advised.***

## 4. Usage
To run the application, simply use the following command:

```bash
python3 app.py
```

When you run the tool, you will see a prompt asking you to provide the file path of the bank statement PDF. You will then choose between two AI models for the extraction process.

**Here is an example terminal output when the tool is run.**

```
**************************************************************
    Hello.
    This is an AI model that parses a bank statement
    uploaded as a PDF file and provides information about it.
**************************************************************

Let's start by locating the PDF file. Without using any quotation marks, please provide the file path. If the file is not in the same directory as the `app.py` file, an absolute path is required:
(default is "test_statement.pdf"): My Bank Statement.pdf

Choose a model.
1) "Qwen2-VL-2B" (lower performance, downloads the model for the first time)
2) "gpt-4o-mini" (higher performance, requires API key but no download)
(default is 2): 2

Output:
{
    "isBankStatement": true,
    "account_holder_name": "Mr Sherlock Holmes",
    "account_holder_address": "221B Baker Street, London, NW1 6XE",
    "account_holder_net_balance": 13579.42,
    "is_modified": false,
    "has_overlay": true,
    "is_template": false
}
```

**Also as a GIF image:**

[![The command output in bash](demo.gif)](https://github.com/mertgulexe/bank_statement/blob/c38441f8d2ce12c3c1c156441984a532a4678b53/demo.gif)

## 5. Code Explanation
### 5.1. Summary
Below is a brief explanation of the main components of the code.
- <code><strong>DataProcess.py</strong></code>: The file contains the output template as a json file, utility functions to create a message template using pre-defined user and system prompts, to parse PDF file, and to clean the desired output.
- <code><strong>ModelVault.py</strong></code>: This code consists of two classes to be used for initiating the AI models:
    - **Qwen2-VL-2B**: Uses the `qwen_chat_template` and instantiates `QwenModel`.
    - **gpt-4o-mini**: Uses the `openai_chat_template`. An API key is asked only once and locally saved for future use after it is provided.
- <code><strong>app.py</strong></code>: The script starts by welcoming the user, and requesting the file path for the PDF file to be parsed. Then, the user is prompted to select a model. The selected model parses the PDF file by the help of utility functions and shows the desired JSON output on the terminal screen.

### 5.2. Details
<details>
    <summary><code><strong>DataProcess.py</strong></code></summary>
    <ol>
        <li> Define a variable for a temporary environment file to be used after fetching OpenAI API key.
        <li> Create a user prompt for parsing the PDF file thoroughly.
            <ul>
                <li> Note that <code>bank_name</code> and <code>bank_address</code> are added to help the model to distinguish the customer/business and bank details. They are not used in the final output.
            </ul>
        <li> Define the JSON format to be used as an output.
        <li> Define the prompt for model to use the metadata.
        <li> Define the system prompt for model to come up with a proper JSON format without any additional pleasantries or greetings.
        <li> Create a function to clean the models' markdown formatted outputs. Also, if the PDF file is not a bank statement, it returns the empty JSON template defined before. Also, checks whether the final output is a JSON format or not.
        <li> Create a function to parse the PDF file and convert it to image to be fed into the models. It also returns the metadata of the PDF file to enrich the user prompt.
        <li> Create a function to extract the metadata. The dates in the metadata are converted to human-readable format for ease of use.
        <li> Create a function to return chat template for Qwen model, which uses the pre-defined prompts, converted image and its related metadata.
        <li> Create a function to return chat template for OpenAI model, which uses the pre-defined prompts, converted <code>base64</code> image and its related metadata.
    </ol>
</details>

<details>
    <summary><code><strong>ModelVault.py</strong></code></summary>
    <ol>
        <li> Define a class for Qwen model initiation.
            <ul>
                <li> Initiates the model configurations after getting the model name.
                <li> The processor consumes the input (prompts and the image), tokenizes and feeds them into the model. The final output is cleaned and returned.
            </ul>
        <li> Define a class for OpenAI model initiation.
            <ul>
                <li> It initiates the client object after getting the model name.
                <li> The client object consumes the input (prompts and the image), gets the model name and generates the answer. The final output is cleaned and returned.
            </ul>
    </ol>
</details>

<details>
    <summary><code><strong>app.py</strong></code></summary>
    <ol>
        <li> Checks if the OpenAI API key is defined before or not.
        <li> Greets the user.
        <li> Takes the file path to the PDF file.
        <li> Asks for the model to be used.
        <li> Initiates the model object.
            <ul>
                <li> If the file is not a PDF or does not exist, throws error while defining <code>messages</code> variable and ends the session.
                <li> If the OpenAI model is initiated, checks whether the related API key is saved. Asks if not, and saves it locally.
                <li> Ends the session if an irrelevant model number is entered.
            </ul>
        <li> Generates the JSON output.
        <li> Removes unnecessary keys.
        <li> Prints the final output.
    </ol>
</details>

## 6. Limitations
- Processes only one page of a PDF file due to local GPU restrictions.
- May exhibit lower performance since the models are not fine-tuned accordingly.
- Due to not being containerised, you may encounter bugs in your personal development environment.
- Does not work on Windows OS.
