import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # small to keep demo quick
DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

PROMPT = (
                "Please clean up the following text by removing all HTML tags and any other unnecessary elements. "
                "The final output should preserve the original meaning, but be formatted using standard English grammar and punctuation. "
                "It should be a single paragraph with no line breaks. "
                "Most importantly, change as little as possible to make the text make sense.\n "
                "EXTREMLY IMPORTANTLY - DO NOT OUTPUT ANYTHING BUT THE CLEANED VERSION (NO INTRODUCTION, NO OTHER LITTLE EXPLENATION), JUST THE CLEANED PARAGRAPH. \n"
                "The text is: <p>What data leads you to believe your odometry is fine?  What does fine mean when related to odometry?  What <span class='abbreviation'>PID<template class='tooltiptext' data-text='Proportional Integral Derivative'></template></span> is fine?  Your wheel azimuth <span class='abbreviation'>PID<template class='tooltiptext' data-text='Proportional Integral Derivative'></template></span>?  Your wheel speed <span class='abbreviation'>PID<template class='tooltiptext' data-text='Proportional Integral Derivative'></template></span>?  Your PathPlanner holonomic controller PIDs?  What data leads you believe all of those are fine, and what does fine mean for <span class='abbreviation'>PID<template class='tooltiptext' data-text='Proportional Integral Derivative'></template></span>?</p><p>Are your vision pose estimates consistent from both cameras in all areas of the field? How consistent are vision pose estimates with the odometry pose estimate if vision correction of odometry pose is turned off?  It looks like you are using fixed standard deviation of your vision results equal to odometry standard deviation for translation.  That might be a bit aggressive on vision confidence, especially farther away from tags, seeing just a single tag, and in cases of higher ambiguity.</p>"
            )

def old_way(prompt: str, runs: int = 3):
    """
    Loads model/tokenizer fresh each call (your original approach).
    """
    print("\n=== Old Way ===")
    times = []
    for i in range(runs):
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            trust_remote_code=True
        ).to(DEVICE)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=32)
        times.append(time.time() - start)
        print(f"Run {i+1}: {times[-1]:.2f} sec")
    print(f"Average: {sum(times)/len(times):.2f} sec")

def optimized_way(prompt: str, runs: int = 3):
    """
    Loads model/tokenizer once, reuses them.
    """
    print("\n=== Optimized Way ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        trust_remote_code=True
    ).to(DEVICE)
    times = []
    for i in range(runs):
        start = time.time()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=32)
        times.append(time.time() - start)
        print(f"Run {i+1}: {times[-1]:.2f} sec")
    print(f"Average: {sum(times)/len(times):.2f} sec")

if __name__ == "__main__":
    old_way(PROMPT, runs=3)
    optimized_way(PROMPT, runs=5)
