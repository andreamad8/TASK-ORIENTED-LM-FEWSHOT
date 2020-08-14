from collections import defaultdict
import pprint
import random
import json
pp = pprint.PrettyPrinter(indent=2)
random.seed(0)


def get_false_example_DST(dic,clas,num): 
    sent_false = []
    sent_class = [e[0] for e in dic[clas]]
    for k,v in dic.items():
        if(k!=clas):
            cnt = 1000
            for sent in v:
                if(sent[0] not in sent_class):
                    sent_false.append([sent[0],'none'])
                    cnt -= 1
                if(cnt == 0):break
    random.shuffle(sent_false)
    return sent_false[:num]

def MWOZ_getDST(args):
    train = json.load(open(f".data/MWoZ_2.1/DST/DST_TRAIN.json"))
    test = json.load(open(f".data/MWoZ_2.1/DST/DST_TEST.json"))
    print("LOADING MWOZ DST")
    train_shots = {}
    for dom, slot in train.items():
        print(dom)
        temp = defaultdict(list)
        for k, v in slot.items():
            for val in v[:int(args.shots/2)]:
                temp[k].append([val[0],val[1]])
            temp[k] += get_false_example_DST(slot,k,int(args.shots/2))
            # print(k,len(temp[k]))
        train_shots[dom] = temp
        # for k,v in temp.items():
        #     print(k,v)
    return train_shots,test
# MWOZ_getDST()


def get_false_example(dic,clas,num): 
    sent_false = []
    sent_class = [e['sys'] for e in dic[clas]]
    for k,v in dic.items():
        if(k!=clas):
            cnt = 1000
            for sent in v:
                if(sent['sys'] not in sent_class):
                    sent_false.append([sent['sys'],'false'])
                    cnt -= 1
                if(cnt == 0):break
    random.shuffle(sent_false)
    return sent_false[:num]

def MWOZ_getACT(args):
    train = json.load(open(f".data/MWoZ_2.1/ACT/ACT_TRAIN.json"))
    test = json.load(open(f".data/MWoZ_2.1/ACT/ACT_TEST.json"))
    print("LOADING MWOZ ACT")
    train_shots = {}
    for dom, act in train.items():
        print(dom)
        temp = defaultdict(list)
        for k, v in act.items():
            for val in v[:int(args.shots/2)]:
                temp[k].append([val['sys'],"true"])
            if(k in ["offerbooked","offerbook"] and args.shots > 25):
                temp[k] += get_false_example(act,k,int(args.shots/2)-5)
            else:
                temp[k] += get_false_example(act,k,int(args.shots/2))
            # print(k,len(temp[k]))
        train_shots[dom] = temp
    
    return train_shots,test

# MWOZ_getACT()