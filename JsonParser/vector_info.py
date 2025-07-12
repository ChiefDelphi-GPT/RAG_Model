import argparse
import json
import datetime as dt
import numpy as np
from math import sqrt
from sentence_transformers import SentenceTransformer

MAC = True
DEBUG = True
vectors = [] # [("cooked", "topic_id", "topic_slug")]

def diff_days(other_date, to_date=dt.datetime.today().date()):
    return (to_date - other_date).days

def extractFeatures(data):
    global vectors
    posts = data["data"]["post_stream"]["posts"]
    q_a = None
    replies = []
    for i, post in enumerate(posts):
        if (i == 0): #evaluate the question
            print()
            print("DOING QUESTION")
            difference = diff_days(dt.date(int(post["created_at"].split("T")[0].split("-")[0]),
                int(post["created_at"].split("T")[0].split("-")[1]),
                int(post["created_at"].split("T")[0].split("-")[2])))
            recencyScore = np.exp(-1.0 * (difference) / 1080)
            confidenceScore = sqrt(post["readers_count"])
            q_a = ([post["cooked"], post["topic_id"], post["topic_slug"]], recencyScore * confidenceScore)
            if DEBUG:
                print(f"QA: {q_a}")
        else:
            replies.append(post)
    print()
    print("DOING REPLIES")
    replies = scoreReplies(replies)
    if DEBUG:
        print()
        print()
        print()
        print("QUESTIONS")
        print(q_a)
        print()
        print()
        print("REPLIES")
        print(replies)
    return (q_a, replies)

def scoreReplies(replies):
    #getting number of positive reactions
    positive_ids = {"heart", "point_up", "+1", "laughing", "call_me_hand", "hugs"}
    negative_ids = {"-1", "question", "cry", "angry"}
    total_positive = sum(
        reaction.get("count")
        for reply in replies
        for reaction in reply["reactions"]
        if reaction.get("id") in positive_ids
    )
    total_negative = sum(
        reaction.get("count")
        for reply in replies
        for reaction in reply["reactions"]
        if reaction.get("id") in negative_ids
    )
    total_reads = sum(
        reply["reads"] for reply in replies
    )
    total_created_at = sum(
        diff_days(dt.date(int(reply["created_at"].split("T")[0].split("-")[0]),
                int(reply["created_at"].split("T")[0].split("-")[1]),
                int(reply["created_at"].split("T")[0].split("-")[2])))  for reply in replies
    )
    if DEBUG:
        print("REPLY STATS")
        print("Total_pos_neg:", total_positive+total_negative)
        print("Total_reads:", total_reads)
        print("Total_created_at:", total_created_at)
    replies_data = []
    best_reps = []
    for reply in replies:
        if reply["accepted_answer"] or reply["topic_accepted_answer"]:
            if DEBUG:
                print("A reply that was the accepted answer:", reply["cooked"])
            best_reps.append(reply)
        local_positive = sum(
            reaction.get("count", 0)
            for reaction in reply["reactions"]
            if reaction.get("id") in positive_ids
        )
        local_negative = sum(
            reaction.get("count", 0)
            for reaction in reply["reactions"]
            if reaction.get("id") in negative_ids
        )
        if DEBUG:
            print("STATS USED FOR SCORE")
            print("Local created at:", diff_days(dt.date(int(reply["created_at"].split("T")[0].split("-")[0]),
                int(reply["created_at"].split("T")[0].split("-")[1]),
                int(reply["created_at"].split("T")[0].split("-")[2]))))
            print("Local reads:", reply["reads"])
            print("Local positive (versus all):", local_positive)
            print("Local negative (versus all):", local_negative)
            
        if total_positive + total_negative == 0:
            score = (0.5*diff_days(dt.date(int(reply["created_at"].split("T")[0].split("-")[0]),
                int(reply["created_at"].split("T")[0].split("-")[1]),
                int(reply["created_at"].split("T")[0].split("-")[2])))/total_created_at) +\
                (0.5*reply["reads"]/total_reads)
        else:
            score = (0.35*diff_days(dt.date(int(reply["created_at"].split("T")[0].split("-")[0]),
                int(reply["created_at"].split("T")[0].split("-")[1]),
                int(reply["created_at"].split("T")[0].split("-")[2])))/total_created_at) +\
                (0.35*reply["reads"]/total_reads)+\
                (0.15*local_positive/(total_positive+total_negative))-\
                (0.15*local_negative/(total_positive+total_negative)) #possibly add score and trust_level
        
        if DEBUG:
            print("Score:", score)
            print("The reply:", reply["cooked"])

        reply_touple = (reply["cooked"], score)
        replies_data.append(reply_touple)
    parsed_replies_data = []
    for i, reply_data in enumerate(replies_data):
        if i == 0:
            parsed_replies_data.append(reply_data)
        if i < 5:
            if reply_data[1] < parsed_replies_data[0][1]:
                parsed_replies_data.append(reply_data)
            else:
                temp = [reply_data]
                parsed_replies_data = temp + parsed_replies_data
        if reply_data[1] < max(x[1] for x in parsed_replies_data):
            parsed_replies_data[0] = reply_data
    replies = [reply[0] for reply in parsed_replies_data]
    return replies

def main(args):
    filename = args.files[0]
    name = filename.split('.json')[0]
    inputFileName = name+'.json'
    if MAC:
        with open(inputFileName, 'r') as inputFile:
            data = json.load(inputFile)
    else:
        with open(inputFileName, 'r', encoding='utf-8') as inputFile:
            data = json.load(inputFile)
    
    data = extractFeatures(data)

    #possibly do something to put the output into a file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON Parser")
    parser.add_argument('files', type=str, nargs='+', help='Input Clened JSON')
    args = parser.parse_args()
    main(args)
