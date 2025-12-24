import argparse
import json
import os
import re

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("ERROR: bs4 (BeautifulSoup) is required. Install with: pip install beautifulsoup4")
    raise

DEBUG = True
MAC = True
SSH = False
LINUX = False
PRINT_MUCH = True

def clean_html(raw_html: str) -> str:
    """Remove HTML tags, scripts, and styles, keeping text only and preserving paragraphs."""
    soup = BeautifulSoup(raw_html, "html.parser")

    for bad in soup(["script", "style"]):
        bad.decompose()

    text_parts = []
    for elem in soup.find_all(["p", "div", "li", "br"]):
        stripped = elem.get_text(" ", strip=True)
        if stripped:
            text_parts.append(stripped)

    if not text_parts:
        whole = soup.get_text(" ", strip=True)
        return re.sub(r"\s+", " ", whole).strip()

    return "\n\n".join(text_parts)

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
                print("Original text (raw HTML-like):", string_org[:200].replace("\n", " ") + ("..." if len(string_org) > 200 else ""))

            baseline_clean = clean_html(string_org)
            if DEBUG:
                print("Baseline cleaned:", baseline_clean.replace("\n", " "))

            post["cooked"] = baseline_clean

    return data

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

    if filename.endswith('.json'):
        name = filename[:-5]
        inputFileName = filename
    else:
        name = filename
        inputFileName = name + '.json'

    data = None
    content = None

    if PRINT_MUCH:
        print(f"Input file: {inputFileName}")

    try:
        if MAC or LINUX:
            with open(inputFileName, 'r') as inputFile:
                content = inputFile.read()
        else:
            with open(inputFileName, 'r', encoding='utf-8') as inputFile:
                content = inputFile.read()

        if PRINT_MUCH:
            print(f"File read successfully (length: {len(content)} chars)")

        try:
            data = json.loads(content)
            if isinstance(data, str):
                if PRINT_MUCH:
                    print("Data is a JSON string, processing...")
                print("File contains JSON string, processing with process_json_string...")
                processed_content = process_json_string(data)
                data = json.loads(processed_content)
        except json.JSONDecodeError:
            if PRINT_MUCH:
                print("Initial JSON parsing failed, treating as raw string...")
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
    elif LINUX:
        outputFileName = "/mnt/c/Users/serge/Downloads/FRC/RAG_Model/Cheif_Delphi_JSONS/" + name.split("/")[-1] + '.json'
    else:
        file_name = os.path.splitext(os.path.basename(name))[0]
        outputFileName = os.path.join(
            r"C:\Users\serge\Downloads\FRC\RAG_model\Cheif_Delphi_JSONS",
            f"{file_name}.json"
        )

    if PRINT_MUCH:
        print(f"Writing to: {outputFileName}")

    if DEBUG:
        print(f"Output written to {outputFileName}")
        print("Data = \n", data)

    if MAC or LINUX:
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