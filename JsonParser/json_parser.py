#opening files
import argparse
import re
import html

DEBUG = True

def getTextFromLine(line):
    start = line.find("\"cooked\": \"<p>") + len("\"cooked\": \"<p>")
    end = line.find("</p>\",")-1
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
            start_abbrev = line.find("<span class=\"abbreviation\">", start)+len("<span class=\"abbreviation\">")
            end_abbrev = line.find("<template class=", start_abbrev)-1
            abbrevText = line[start_abbrev:end_abbrev]
            line = line.replace(spanText, f"{abbrevText} ({realText})")
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
                outputFileName = "/Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/JsonParser/debug_output.txt"
                with open(outputFileName, 'a') as debug_file:
                    debug_file.write(f"Original: {string_org}\n")
                    debug_file.write(f"Cleaned: {string_new}\n\n")
                print(f"Line {i}:", line)
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

    if (DEBUG == True):
        print(f"Output written to {outputFileName}")
        # for i, line in enumerate(lines):
        #     print(f"Line {i}:\t", line)
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