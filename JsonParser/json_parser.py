#opening files
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

DEBUG = True

def getTextFromLine(line):
    start = line.find("\"cooked\": \"<p>") + len("\"cooked\": \"<p>")
    end = line.find("</p>',")-len("</p>',")
    return line[start:end]

def queryDeepSeek(input_text):
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"   # Smallest - fastest loading
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"    # Good balance of quality and resource usage
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"   # Alternative 8B option
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"   # Larger for better reasoning
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"   # High-end consumer hardware
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"  # Very large - needs lots of RAM
    # model_name = "deepseek-ai/DeepSeek-R1"                     # Main flagship - enterprise only (671B)

    if (torch.backends.mps.is_available()):
        device = "mps"
        torch_dtype = torch.float32  # MPS works better with float32
    elif (torch.cuda.is_available()):
        device = "cuda"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=None,  # Don't use auto device mapping
        trust_remote_code=True  # DeepSeek models may require this
    ).to(device)

    messages = [
        {"role": "user", "content": input_text}
    ]
    try: 
        formatted_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except:
        formatted_input = f"User: {input_text}\nAssistant:"

    start_time = time.time()
    input_ids = tokenizer.encode(formatted_input, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10000,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
        )

    response_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    if DEBUG:
        print()
        print("MODEL RESPONSE:")
        print(response.strip())
        print()
    end_time = time.time()
    return response.strip(), end_time - start_time

def cleanText(lines):
    for i, line in enumerate(lines):
        if ("\"cooked\": \"" in line):
            string_org = getTextFromLine(line)
            prompt = (
                "Please clean up the following text by removing all HTML tags and any other unnecessary elements. "
                "The final output should preserve the original meaning, but be formatted using standard English grammar and punctuation. "
                "It should be a single paragraph with no line breaks. "
                "The text is: " + string_org
            )
            string_new, time_taken = queryDeepSeek(prompt)
            line[i] = line.replace(string_org, string_new)
            if DEBUG:
                print()
                print()
                print()
                print("Processing line:")
                print("Original text:", string_org)
                print()
                print("Cleaned text:", string_new)
                print()
                print(f"Time taken for this line: {time_taken:.2f} seconds")
    return lines

def main(args):
    filename = args.files[0]
    name = filename.split('.json')[0]
    inputFileName = name+'.json'
    inputFile = open(inputFileName, 'r')
    inputText = inputFile.read()
    lines = inputText.splitlines()
    #DO THINGS WITH THE INPUT TEXT IN FUNCTIONS
    lines = cleanText(lines)
    outputFileName = "/Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/JsonParser/"+ name.split("/")[-1] + '_output.txt'
    outputFile = open(outputFileName, 'w')


    if DEBUG:
        print(f"Output written to {outputFileName}")
        # for i, line in enumerate(lines):
        #     print(f"Line {i}:\t", line)
    else:
        outputFile.write('\n'.join(lines)) # MAYBE CHANGE
    inputFile.close()
    outputFile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JSON Parser')
    parser.add_argument('files', type=str, nargs='+', 
                        help='Input JSON filename')
    args = parser.parse_args()
    main(args)
