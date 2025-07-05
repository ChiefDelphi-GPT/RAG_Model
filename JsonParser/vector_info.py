import argparse
import json
import datetime as dt

MAC = True
DEBUG = False
vectors = [] #the first element of this dictionary is the question and the rest are the answers

def extractFeatures(data):
    global vectors
    posts = data["data"]["post_stream"]["posts"]
    q_a = ()
    replies = []
    for i, post in enumerate(posts):
        if (i == 1): #evaluate the question
            q_a = ([post["cooked"], post["topic_id"], post["topic_slug"]], [post["created_at"], post["readers_count"], post["trust_level"]])
        replies.append(post)
        if DEBUG:
            print("new post")
            print(post)
            print()
            print()
            print()
            print()
            print()
    replies = scoreReplies(replies)

def scoreReplies(replies):
    #getting number of positive reactions
    positive_ids = {"heart", "point_up", "+1", "laughing", "call_me_hand", "hugs"}
    negative_ids = {"-1", "question", "cry", "angry"}
    total_positive = sum(
        reaction.get("count", 0)
        for reaction in replies.get("reactions", [])
        if reaction.get("id") in positive_ids
    )
    total_negative = sum(
        reaction.get("count", 0)
        for reaction in replies.get("reactions", [])
        if reaction.get("id") in negative_ids
    )
    total_reads = sum(
        reply["reads"] for reply in replies
    )
    total_trust_level = sum(
        reply["truts_level"] for reply in replies
    )
    total_scores = sum(
        reply["score"] for reply in replies
    )
    total_created_at = sum(
        diff_days(dt.date(reply["created_at"].split("T")[0].split("-")[0],
                reply["created_at"].split("T")[0].split("-")[1],
                reply["created_at"].split("T")[0].split("-")[2]))  for reply in replies
    )
    replies_data = []
    best_reps = []
    for reply in replies:
        if reply["accepted_answer"] or reply["topic_accepted_answer"]:
            best_reps.append(reply)
        local_positive = sum(
            reaction.get("count", 0)
            for reaction in reply.get("reactions", [])
            if reaction.get("id") in positive_ids
        )
        local_negative = sum(
            reaction.get("count", 0)
            for reaction in reply.get("reactions", [])
            if reaction.get("id") in negative_ids
        )
        score = (0.2*reply["created_at"]/total_created_at) +\
                (0.3*reply["reads"]/total_reads)+\
                (0.15*local_positive/(total_positive+total_negative))-\
                (0.15*local_negative/(total_positive+total_negative))+\
                (0.1*replies["trust_level"]/total_trust_level)+\
                (0.1*replies["score"]/total_scores)
        reply_touple = (replies["cooked"], score)
        replies_data.append(reply_touple)
    parsed_replies_data = []
    for i, reply_data in enumerate(replies_data):
        if i == 1:
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

def diff_days(other_date, to_date=dt.date(dt.datetime.today().date().split('-')[0],
                      dt.datetime.today().date().split('-')[1], 
                      dt.datetime.today().date().split('-')[2])):
    return (to_date - other_date).days

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
