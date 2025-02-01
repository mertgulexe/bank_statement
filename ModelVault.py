import os
import warnings
from typing import List
from DataProcess import clean_output, TEMP_ENV_FILE
# OpenAI model imports
from openai import OpenAI, AuthenticationError
# Qwen Model imports
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    # BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    # Qwen2_5_VLForConditionalGeneration,
)


warnings.filterwarnings(action="ignore")


class QwenModel:
    """
    A Visual Question Answering (VQA) model using Qwen2-VL (default) for generation.
    """
    def __init__(
            self,
            model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
            processor_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    ) -> None:
        """
        Initializes the VQAModel with a pre-trained model and processor.
        
        Args:
            model_name (str): Name or path of the pre-trained model.
            processor_name (str): Name or path of the processor.
        """
        # model configurations
        model_config = dict(
            # pretrained_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",  # OOM
            # pretrained_model_name_or_path="cyan2k/Qwen2.5-VL-7B-Instruct-bnb-4bit",  # lower performance
            pretrained_model_name_or_path=model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
            # offload_buffers=True,
            # quantization_config=BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=torch.float16,
            #     bnb_4bit_use_double_quant=True
            # )
        )

        # load the model and processor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            **model_config
        )
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=processor_name
        )

    def generate(
            self,
            messages: List[dict],
            max_new_tokens: int = 256
    ) -> dict:
        """
        Generates a response based on the provided messages and bank statement pdf file.
        
        Args:
            messages (List[dict]): A list of messages in the chat template format.
            max_new_tokens (int): Maximum number of new tokens to generate. Default is 512.
        
        Returns:
            dict: A json response contains information from the bank statement.
        """
        # create the chat template by using messages
        chat_template = self.processor.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # get the inputs (we don't have a video input anyway)
        image_inputs, video_inputs = process_vision_info(
            conversations=messages
        )

        # tokenize the input
        inputs = self.processor(
            text=[chat_template],
            images=image_inputs,
            videos=video_inputs,  # None.
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # generate the output ids
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

        # remove the input from the generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # get the final LLM output
        dirty_json_output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # clean the output
        return clean_output(dirty_json_output)


class OpenAIModel:
    """
    OpenAI's model for Visual Question Answering (VQA) task.
    """
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, messages: List[dict]) -> dict:
        """
        Generates a response based on the provided messages.
        
        Args:
            messages (List[dict]): A list of messages in the chat template format.
        
        Returns:
            dict: A json response contains information from the bank statement.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
        except AuthenticationError as e:
            print("Sorry, the OpenAI API key seems incorrect.")
            os.remove(TEMP_ENV_FILE) if os.path.exists(TEMP_ENV_FILE) else None
            return {}

        clean_response = clean_output(
            dirty_json=response.choices[0].message.content
        )
        return clean_response
