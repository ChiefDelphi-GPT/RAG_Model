import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import json
import os

DEBUG = False
MAC = False
SSH = True

def getTextFromLine(line):
    start = line.find("\"cooked\": \"")+len("'cooked': '")
    end = line.find("</p>\",")
    if DEBUG:
        print("START:", start)
        print("END:", end)
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

    print(f"    Using device: {device}")
    print(f"    Loading model: {model_name}")

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

def cleanText(data, filename):
    # Add defensive checks for data structure
    if not isinstance(data, dict):
        print(f"  ERROR: Expected dictionary but got {type(data)}")
        return data
    
    if "data" not in data:
        print(f"  ERROR: No 'data' key found in JSON structure")
        print(f"  Available keys: {list(data.keys())}")
        return data
    
    if not isinstance(data["data"], dict) or "post_stream" not in data["data"]:
        print(f"  ERROR: No 'post_stream' key found in data structure")
        if isinstance(data["data"], dict):
            print(f"  Available keys in data: {list(data['data'].keys())}")
        return data
    
    if not isinstance(data["data"]["post_stream"], dict) or "posts" not in data["data"]["post_stream"]:
        print(f"  ERROR: No 'posts' key found in post_stream structure")
        if isinstance(data["data"]["post_stream"], dict):
            print(f"  Available keys in post_stream: {list(data['data']['post_stream'].keys())}")
        return data
    
    posts = data["data"]["post_stream"]["posts"]
    
    if not isinstance(posts, list):
        print(f"  ERROR: Expected posts to be a list but got {type(posts)}")
        return data
    
    total_posts = len(posts)
    
    print(f"  Processing {total_posts} posts in {filename}")
    print("  " + "="*50)
    
    processed_posts = 0
    
    for i, post in enumerate(posts):
        if not isinstance(post, dict):
            print(f"    Skipping post {i+1}/{total_posts} (not a dictionary)")
            continue
            
        if "cooked" in post:
            print(f"    Processing post {i+1}/{total_posts}...", end=" ", flush=True)
            
            string_org = post["cooked"]
            if DEBUG:
                print("Original text:", string_org)
            
            prompt = (
                "Please clean up the following text by removing all HTML tags and any other unnecessary elements. "
                "The final output should preserve the original meaning, but be formatted using standard English grammar and punctuation. "
                "It should be a single paragraph with no line breaks. "
                "Most importantly, change as little as possible to make the text make sense. "
                "The text is: " + string_org
            )
            
            model_response, elapsed_time = queryDeepSeek(prompt)
            post["cooked"] = post["cooked"].replace(string_org, model_response[model_response.find('</think>')+len("</think>"):].lstrip())
            
            if post["cooked"].endswith("</p>\","):
                post["cooked"] = post["cooked"].replace("</p>\",", "\",")
            
            processed_posts += 1
            print(f"Done! ({elapsed_time:.2f}s)")
            
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
            print(f"    Skipping post {i+1}/{total_posts} (no 'cooked' field)")
    
    print(f"  Completed processing {processed_posts} posts from {filename}")
    print("  " + "="*50)
    
    return data
    
    return data

def process_json_string(json_str):
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("  ERROR: NOT VALID JSON AT ALL")
        print(f"  Error: {e}")
        raise ValueError(f"Invalid JSON input: {e}")

    cleaned_data = cleanText(data)
    return json.dumps(cleaned_data, indent=4, ensure_ascii=False)

def main(args):
    inputFileName = args.files[0]
    
    print("\n" + "="*80)
    print(f"STARTING PROCESSING FOR FILE: {inputFileName}")
    print(f"DEBUG: Full input path: {inputFileName}")
    print(f"DEBUG: File exists: {os.path.exists(inputFileName)}")
    print(f"DEBUG: Base filename will be: {inputFileName.split('/')[-1].split('.json')[0]}")
    print("="*80)
    
    data = None
    content = None
    
    # Step 1: Read the file
    try:
        print(f"Reading file: {inputFileName}")
        
        if MAC:
            with open(inputFileName, 'r') as inputFile:
                content = inputFile.read()
        else:
            with open(inputFileName, 'r', encoding='utf-8') as inputFile:
                content = inputFile.read()
        
        print("  File read successfully!")
        print(f"  Content length: {len(content)} characters")
        print(f"  First 200 characters: {content[:200]}")
        
    except Exception as e:
        print(f"  ERROR: Error reading file: {e}")
        print("  CANNOT CONTINUE WITHOUT FILE CONTENT - EXITING")
        return
    
    # Step 2: Parse JSON - with detailed diagnostics
    try:
        print("  Attempting to parse as JSON...")
        data = json.loads(content)
        print("  File parsed as JSON successfully!")
        print(f"  Data type after parsing: {type(data)}")
        
        if isinstance(data, dict):
            print(f"  Root dictionary keys: {list(data.keys())}")
        elif isinstance(data, list):
            print(f"  Root is a list with {len(data)} items")
            if len(data) > 0:
                print(f"  First item type: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"  First item keys: {list(data[0].keys())}")
        else:
            print(f"  Root is neither dict nor list, it's: {type(data)}")
            print(f"  Value: {str(data)[:200]}...")
        
    except json.JSONDecodeError as e:
        print(f"  ERROR: Failed to parse JSON: {e}")
        print(f"  JSON Error details: {str(e)}")
        print(f"  Error at position: {e.pos if hasattr(e, 'pos') else 'unknown'}")
        print("  CANNOT CONTINUE WITHOUT VALID JSON - EXITING")
        return
    
    # Step 3: Detailed structure analysis
    print("\n  DETAILED STRUCTURE ANALYSIS:")
    print("  " + "="*50)
    
    structure_ok = False
    
    try:
        if isinstance(data, dict):
            print(f"  ✓ Root is dictionary with keys: {list(data.keys())}")
            
            if "data" in data:
                print(f"  ✓ Has 'data' key, type: {type(data['data'])}")
                
                if isinstance(data["data"], dict):
                    print(f"  ✓ data['data'] is dict with keys: {list(data['data'].keys())}")
                    
                    if "post_stream" in data["data"]:
                        print(f"  ✓ Has 'post_stream', type: {type(data['data']['post_stream'])}")
                        
                        if isinstance(data["data"]["post_stream"], dict):
                            print(f"  ✓ post_stream is dict with keys: {list(data['data']['post_stream'].keys())}")
                            
                            if "posts" in data["data"]["post_stream"]:
                                posts = data["data"]["post_stream"]["posts"]
                                print(f"  ✓ Has 'posts', type: {type(posts)}")
                                
                                if isinstance(posts, list):
                                    print(f"  ✓ posts is list with {len(posts)} items")
                                    if len(posts) > 0:
                                        print(f"  ✓ First post type: {type(posts[0])}")
                                        if isinstance(posts[0], dict):
                                            print(f"  ✓ First post keys: {list(posts[0].keys())}")
                                            structure_ok = True
                                        else:
                                            print(f"  ✗ First post is not dict: {type(posts[0])}")
                                    else:
                                        print("  ! Posts list is empty")
                                        structure_ok = True  # Empty is still valid structure
                                else:
                                    print(f"  ✗ posts is not list: {type(posts)}")
                            else:
                                print(f"  ✗ No 'posts' in post_stream")
                        else:
                            print(f"  ✗ post_stream is not dict: {type(data['data']['post_stream'])}")
                    else:
                        print(f"  ✗ No 'post_stream' in data")
                else:
                    print(f"  ✗ data['data'] is not dict: {type(data['data'])}")
            else:
                print(f"  ✗ No 'data' key in root")
        else:
            print(f"  ✗ Root is not dict: {type(data)}")
            
    except Exception as e:
        print(f"  ERROR during structure analysis: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
    
    print(f"  Structure OK: {structure_ok}")
    print("  " + "="*50)
    
    # Step 4: Attempt processing regardless of structure issues
    if structure_ok:
        print("\n✓ Structure looks good, proceeding with normal processing...")
        try:
            data = cleanText(data, inputFileName)
            print("✓ Text cleaning completed successfully!")
        except Exception as e:
            print(f"ERROR during cleanText: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            print("! Continuing with original data...")
    else:
        print("\n! Structure issues detected, but continuing anyway...")
        print("! Attempting to process with modified approach...")
        
        # Try alternative processing approaches
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], str):
                print("! Detected data['data'] is a string, attempting to parse it...")
                try:
                    nested_data = json.loads(data["data"])
                    print(f"! Successfully parsed nested JSON, type: {type(nested_data)}")
                    if isinstance(nested_data, dict):
                        data["data"] = nested_data
                        print("! Replaced string with parsed object, retrying...")
                        try:
                            data = cleanText(data, inputFileName)
                            print("✓ Text cleaning completed after nested parse!")
                        except Exception as e:
                            print(f"! Still failed after nested parse: {e}")
                except Exception as e:
                    print(f"! Failed to parse nested JSON: {e}")
            else:
                print("! No clear fix available, saving original data...")
        else:
            print("! Data is not a dictionary, saving as-is...")
    
    # Step 5: Always try to write output
    try:
        base_filename = inputFileName.split('/')[-1].split('.json')[0]
        
        if SSH: 
            outputFileName = f"/home/rhayrapetyan/automatic/Cheif_Delphi_JSONS/{base_filename}.json"
        elif MAC:
            outputFileName = f"/Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/Cheif_Delphi_JSONS/{base_filename}.json"
        else:
            outputFileName = rf"C:\Users\serge\Downloads\FRC\RAG_model\Cheif_Delphi_JSONS\{base_filename}.json"
        
        print(f"\nWriting output to: {outputFileName}")
        
        if MAC:
            with open(outputFileName, 'w') as outputFile:
                json.dump(data, outputFile, indent=4, ensure_ascii=False)
        else:
            with open(outputFileName, 'w', encoding='utf-8') as outputFile:
                json.dump(data, outputFile, indent=4, ensure_ascii=False)
        
        print("✓ Output file written successfully!")
        
    except Exception as e:
        print(f"ERROR: Failed to write output file: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        print("! This is a critical error, but file analysis is complete")
    
    print(f"\n📊 ANALYSIS COMPLETED FOR: {inputFileName}")
    print(f"   Structure Valid: {structure_ok}")
    print(f"   Data Type: {type(data)}")
    if isinstance(data, dict):
        print(f"   Root Keys: {list(data.keys())}")
    print("="*80)
    print("\n")