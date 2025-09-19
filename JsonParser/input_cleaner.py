import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import json

DEBUG = True
MAC = False
SSH = True
PRINT_MUCH = True

def getTextFromLine(line):
    start = line.find("\"cooked\": \"")+len("'cooked': '")
    end = line.find("</p>\",")
    if DEBUG:
        print("START:", start)
        print("END:", end)
    return line[start:end]

def queryDeepSeek(input_text):
    if PRINT_MUCH:
        print(f"Starting DeepSeek query (input length: {len(input_text)} chars)...")
    
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"   # Smallest - fastest loading
    # model_name = "google/gemma-2b-it"
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
        if PRINT_MUCH:
            print("Chat template failed, using fallback format")

    start_time = time.time()
    input_ids = tokenizer.encode(formatted_input, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20000,
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
    
    if PRINT_MUCH:
        print(f"DeepSeek query completed in {end_time - start_time:.2f} seconds")
    
    return response.strip(), end_time - start_time

def cleanText(data):
    posts = data["data"]["post_stream"]["posts"]
    
    if PRINT_MUCH:
        print(f"Processing {len(posts)} posts...")
    
    for i, post in enumerate(posts):
        if PRINT_MUCH:
            print(f"Processing post {i+1}/{len(posts)}")
        
        if "cooked" in post:
            string_org = post["cooked"]
            if DEBUG:
                print("Original text:", string_org)
            
            prompt = (
                "Please clean up the following text by removing all HTML tags and any other unnecessary elements. "
                "The final output should preserve the original meaning, but be formatted using standard English grammar and punctuation. "
                "It should be a single paragraph with no line breaks. "
                "Most importantly, change as little as possible to make the text make sense.\n "
                "EXTREMLY IMPORTANTLY - DO NOT OUTPUT ANYTHING BUT THE CLEANED VERSION (NO INTRODUCTION, NO OTHER LITTLE EXPLENATION), JUST THE CLEANED PARAGRAPH. \n"
                "The text is: " + string_org
            )
            
            model_response, elapsed_time = queryDeepSeek(prompt)
            post["cooked"] = post["cooked"].replace(string_org, model_response[model_response.find('</think>')+len("</think>"):].lstrip())
            if post["cooked"].endswith("</p>\","):
                post["cooked"] = post["cooked"].replace("</p>\",", "\",")
            
            if DEBUG:
                print()
                print()
                print()
                print("Original text:", string_org)
                print()
                print("Model response:", model_response)
                print()
                print("Time taken:", elapsed_time, "seconds")
        else:
            if PRINT_MUCH:
                print(f"Post {i+1} has no 'cooked' field, skipping")
    
    return data
    # for i, line in enumerate(lines):
    #     if ("\"cooked\": \"" in line):
    #         string_org = getTextFromLine(line)
    #         print(string_org)
    #         print()
    #         print()
    #         prompt = (
    #             "Please clean up the following text by removing all HTML tags and any other unnecessary elements. "
    #             "The final output should preserve the original meaning, but be formatted using standard English grammar and punctuation. "
    #             "It should be a single paragraph with no line breaks. "
    #             "Importantly, if it seems to make sense don't change anything, just return the text as is. "
    #             "The text is: " + string_org
    #         )
    #         model_response, elapsed_time = queryDeepSeek(prompt)
    #         lines[i] = line.replace(string_org, model_response[model_response.find('</think>')+len("</think>"):].lstrip())
    #         if lines[i].endswith("</p>\","):
    #             lines[i] = lines[i].replace("</p>\",", "\",")
    #         if DEBUG:
    #             print()
    #             print()
    #             print()
    #             print("Original text:", string_org)
    #             print()
    #             print("Model response:", model_response)
    #             print()
    #             print("Time taken:", elapsed_time, "seconds")
    # return lines

def process_json_string(json_str):
    if PRINT_MUCH:
        print(f"Processing JSON string (length: {len(json_str)} chars)...")
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("NOT VALID JSON AT ALL")
        print()
        print(f"Error: {e}")
        print()
        print()
        print()
        raise ValueError(f"Invalid JSON input: {e}")

    cleaned_data = cleanText(data)
    return json.dumps(cleaned_data, indent=4, ensure_ascii=False)

def main(args):
    filename = args.files[0]
    
    if PRINT_MUCH:
        print(f"\n{'='*60}")
        print(f"PROCESSING FILE: {filename}")
        print(f"{'='*60}")
    
    # Fix the filename parsing - don't remove .json extension
    if filename.endswith('.json'):
        name = filename[:-5]  # Remove .json extension
        inputFileName = filename  # Use the original filename
    else:
        name = filename
        inputFileName = name + '.json'
    
    data = None
    content = None
    
    if PRINT_MUCH:
        print(f"Input file: {inputFileName}")
    
    try:
        if MAC:
            with open(inputFileName, 'r') as inputFile:
                content = inputFile.read()
        else:
            with open(inputFileName, 'r', encoding='utf-8') as inputFile:
                content = inputFile.read()
        
        if PRINT_MUCH:
            print(f"File read successfully (length: {len(content)} chars)")
        
        try:
            data = json.loads(content)
            # Check if data is actually a string (JSON containing a string)
            if isinstance(data, str):
                if PRINT_MUCH:
                    print("Data is a JSON string, processing...")
                print("File contains JSON string, processing with process_json_string...")
                processed_content = process_json_string(data)
                data = json.loads(processed_content)
        except json.JSONDecodeError:
            if PRINT_MUCH:
                print("Initial JSON parsing failed, treating as raw string...")
            # If it's not valid JSON, treat it as a raw string and process it
            print("File contains raw string, processing with process_json_string...")
            processed_content = process_json_string(content)
            data = json.loads(processed_content)
        
        if isinstance(data["data"], str):
            if PRINT_MUCH:
                print("Parsing nested JSON in data['data']...")
            data["data"] = json.loads(data["data"])
            
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        if PRINT_MUCH:
            print(f"ERROR: File {inputFileName} not found")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        if PRINT_MUCH:
            print(f"ERROR reading {inputFileName}: {e}")
        return
    
    print()
    print()
    print()
    print()
    print("FILENAME:", inputFileName)
    print("TYPE:", type(data))
    print("TYPE data[data]:", type(data["data"]))
    print("TYPE data[data]:", type(data["data"]))
    
    data = cleanText(data)
    
    if SSH: 
        outputFileName = "/home/rhayrapetyan/automatic/Cheif_Delphi_JSONS/" + name.split("/")[-1] + ".json"
    elif MAC:
        outputFileName = "/Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/Cheif_Delphi_JSONS/"+ name.split("/")[-1] + '.json'
    else:
        file_name = name.split("/")[-1]
        outputFileName = rf"C:\Users\serge\Downloads\FRC\RAG_model\Cheif_Delphi_JSONS\{file_name}.json"
    
    if PRINT_MUCH:
        print(f"Writing to: {outputFileName}")
    
    if DEBUG:
        print(f"Output written to {outputFileName}")
        print("Data = \n", data)

    if MAC:
        with open(outputFileName, 'w') as outputFile:
            json.dump(data, outputFile, indent=4, ensure_ascii=False)
    else:
        with open(outputFileName, 'w', encoding='utf-8') as outputFile:
            json.dump(data, outputFile, indent=4, ensure_ascii=False)

    if PRINT_MUCH:
        print(f"COMPLETED: {filename}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON Parser")
    parser.add_argument('files', type=str, nargs='+', help='Input Clened JSON')
    args = parser.parse_args()
    main(args)