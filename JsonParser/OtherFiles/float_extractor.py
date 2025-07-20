import re

def extract_float_in_range(text):
    """
    Extract a floating point number between 0.1 and 4.1 from a string.
    
    Args:
        text (str): Input string containing a floating point number
        
    Returns:
        float: The floating point number if found and in range, None otherwise
    """
    # Regex pattern to match floating point numbers
    # This matches: optional minus, digits, optional decimal point and more digits
    pattern = r'-?\d+\.?\d*'
    
    # Find all potential numbers in the string
    matches = re.findall(pattern, text)
    
    for match in matches:
        try:
            # Convert to float
            num = float(match)
            # Check if it's in the specified range
            if 0.1 <= num <= 4.1:
                return num
        except ValueError:
            # Skip if conversion fails
            continue
    
    return None

# Test with your example
test_string = "hi bo.jlkadsf;s.;sdalkf 1.asl;dkjfeoweifj4.0"
result = extract_float_in_range(test_string)
print(f"Input: {test_string}")
print(f"Output: {result}")

# Additional test cases
test_cases = [
    "hello 2.5 world",
    "value is 0.1 exactly",
    "maximum 4.1 allowed",
    "too high 5.0 rejected",
    "negative -1.0 rejected", 
    "multiple 1.2 and 3.4 numbers",
    "no numbers here",
    "edge case 0.09 too small",
    "another edge 4.11 too big"
]

print("\nAdditional test cases:")
for test in test_cases:
    result = extract_float_in_range(test)
    print(f"'{test}' -> {result}")