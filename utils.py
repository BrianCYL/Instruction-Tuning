from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    # return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"
    # return f"你是人工智慧助理，並且精通文言文及白話文的雙向翻譯，以下是用戶和人工智慧助理之間的對話。你要對用戶的問題提供有用、精準且正確的回答。USER: {instruction} ASSISTANT:"
    return f"你是人工智慧助理，並且精通文言文及白話文的雙向翻譯，以下是用戶和人工智慧助理之間的對話，內容會有關白話文及文言文的翻譯問題。你要對用戶的問題提供有用、精準且正確的回答。USER: {instruction} ASSISTANT:"

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config

def get_fewshots(instruction, data, index, cnt):
    if cnt == 0:
        return f"你是人工智慧助理，並且精通文言文及白話文的雙向翻譯，以下是用戶和人工智慧助理之間的對話，內容會有關白話文及文言文的翻譯問題。你要對用戶的問題提供有用、精準且正確的回答。USER: {instruction} ASSISTANT:"
    else:
        # Get the valid indices
        avail_idx = torch.arange(len(data))
        valid_idx = avail_idx[avail_idx != index]
        # Randomly sample from the valid indices
        random_ids = valid_idx[torch.randint(0, len(valid_idx), (cnt,))].tolist()

        example = "以下是你之前協助用戶進行翻譯的範例：\n"
        for idx in random_ids:
            example += f"USER:{data[idx]['instruction']} ASSISTANT:{data[idx]['output']} \n"
        example += "以下是你需要協助用戶進行翻譯的問題，請參考先前提供的範例，提供正確簡潔的答案。\n"
        return f"你是人工智慧助理，並且精通文言文及白話文的雙向翻譯，以下是用戶和人工智慧助理之間的對話，內容會有關白話文及文言文的翻譯問題。你要對用戶的問題提供有用、精準且正確的回答。{example}\nUSER: {instruction} ASSISTANT:" 
