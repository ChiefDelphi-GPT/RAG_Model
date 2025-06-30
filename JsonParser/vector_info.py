import argparse
import json

MAC = True
DEBUG = False
vectors = [] #the first element of this dictionary is the question and the rest are the answers
topic-slug_set = {}

def extractFeatures(data):
    global vectors
    posts = data["data"]["post_stream"]["posts"]
    q_a = {}
    replies = []
    for i, post in enumerate(posts):
        if (i == 1): #evaluate the question
            # q_a[post["cooked"]] = [
            #     post["created_at"],
            #     post["readers_count"],
            #     post["trust_level"]
            # ]
            q_a[(post["cooked"], post["topic_id"], post["topic_slug"])] = [
                post["created_at"],
                post["readers_count"],
                post["trust_level"]
            ]
        replies.append(post)
        if DEBUG:
            print("new post")
            print(post)
            print()
            print()
            print()
            print()
            print()
    scoreReplies(data, replies)

def scoreReplies(data, replies):
    #getting number of positive reactions
    positive_ids = {"heart", "point_up", "+1", "laughing", "call_me_hand", "hugs"}
    negative_ids = {"-1", "question", "cry", "angry", ""}
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
    replies_data = []
    for reply in replies:
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
        reply_dict = {}
        reply_dict[replies["cooked"]] = [
            replies["accepted_answer"],
            replies["topic_accepted_answer"],
            replies["created_at"],
            replies["reads"],
            positive_ids/(total_positive+total_negative),
            negative_ids/(total_positive+total_negative),
            replies["trust_level"],
            replies["score"]
        ]
        replies_data.append()

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
