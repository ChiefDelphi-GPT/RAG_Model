#opening files
import argparse
import re
import html

DEBUG =- True

def getTextFromLine(line):
    start = len("\"cooked\": \"")
    end = line.find(",")-1
    return line[start:end]

def fixFromSpanClass(line):
    while ("<span class=\"" in line):
        start = line.find("<span class=\"")
        end = line.find("</span>", start)+len("</span>")
        spanText = line[start:end]
        if ("<span class=\"abbreviation\">" in spanText):
            start_real = line.find("data-text=\"", start)+len("data-text=\"")
            end_real = line.find("\"", start_real)
            realText = line[start_real:end_real]
            abbr_match = re.search(r'<span class="[^"]*">(.*?)<template', spanText, re.DOTALL)
            if abbr_match:
                abbrText = abbr_match.group(1).strip()
                replacement = f"{abbrText} ({realText})"
                line = line.replace(spanText, replacement)
            else:
                line = line.replace(spanText, realText)
    return line
        #add other cases of span classes as we go

def cleanText(lines):
    for i, line in enumerate(lines):
        if ("\"cooked\": \"" in line):
            string_org = getTextFromLine(line)
            string_new = fixFromSpanClass(string_org)
            string_new = re.sub(r'<p>', '\n', string_new)
            string_new = re.sub(r'</p>', '\n', string_new)
            string_new = re.sub(r'<br\s*/?>', '\n', string_new)
            string_new = re.sub(r'<[^>]+>', '', string_new)
            string_new = html.unescape(string_new)
            string_new = re.sub(r'\n+', '\n', string_new).strip()
            lines[i].replace(string_org, string_new)
            if DEBUG:
                print(f"Original: {string_org}")
                print(f"Cleaned: {string_new}")
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
    outputFileName = "/Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/JsonParser/" + name.split("/")[-1] + '_output.txt'

    if (DEBUG == True):
        print(f"Output written to {outputFileName}")
    else:
        outputFile = open(outputFileName, 'w')
        outputFile.write('\n'.join(lines)) # MAYBE CHANGE
        inputFile.close()
        outputFile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JSON Parser')
    parser.add_argument('files', type=str, nargs='+', 
                        help='Input JSON filename')
    args = parser.parse_args()
    main(args)