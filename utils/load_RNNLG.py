import json
import random

def load_file(domain,split):
    fin = open(f".data/RNNLG/{domain}/{split}.json")
    # fin = open(f".data/RNNLG/original/{domain}/{split}.json")
    # remove comment lines
    # for _ in range(5):
    #     fin.readline()
    dat = json.load(fin)
    fin.close()
    return dat

# def NLG():
def RNNLG(args,domain):
    if(domain in ["tv","laptop"]):
        train = load_file(domain,"train")[:10]
    else:
        train = load_file(domain,"train")[:args.shots]
    test = load_file(domain,"test")

    return train, test

# domains = ["hotel","laptop","restaurant","tv"]
# for d in domains:
#     RNNLG(d)
# # NLG()


