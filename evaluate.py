
from utils.conll2002_metrics import *
from tabulate import tabulate
from sklearn.metrics import f1_score, accuracy_score
from models.SC_GPT.evaluator import score
import json
import os
import numpy as np
import subprocess
import tempfile
import editdistance
import random
import matplotlib.pyplot as plt
import seaborn as sns

def eval_f1(filename):
    # print("========================================")
    # print("evaluating %s" % filename)
    try:
        json_file = open(filename, "rb")
    except:
        # print("%s file doesn't exist !" % filename)
        return 0.0

    data = json.load(json_file)
    data = data["results"]
    data_indices = list(data.keys())

    lines = []
    for index in data_indices:
        data_item = data[index]
        preds = data_item["PRED"]
        text = data_item["text"]
        token_list = text.split()
        label_list = data_item["TAGS"]
        pred_list = ["O"] * len(label_list)
        for slot_type, slot_value in preds.items():
            if slot_value == "none":
                continue
            if slot_value in text:
                values_token = slot_value.split()
                for i, token in enumerate(values_token):
                    try:
                        ind = token_list.index(token)
                        if i == 0:
                            pred_list[ind] = "B-" + slot_type
                        else:
                            pred_list[ind] = "I-" + slot_type
                    except:
                        break
        
        for token, pred, label in zip(token_list, pred_list, label_list):
            lines.append(token + " " + pred + " " + label)
    
    json_file.close()
    results = conll2002_measure(lines)
    f1 = results["fb1"]
    # print("f1 socre: %.4f" % f1)
    return f1
    

classes = ["addtoplaylist","bookrestaurant","getweather","playmusic","searchscreeningevent","ratebook","searchcreativework"]


def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]


def get_closest_edit(elm):
    return argmin([editdistance.eval(elm, c) for c in classes])

def get_f1(file):
    json_file = json.load(open(filename))
    y_true = []
    y_pred = []
    for _,elm in json_file["results"].items():
        y_pred.append(get_closest_edit(elm["PRED"]))
        y_true.append(classes.index(elm["GOLD"]))
    macro = f1_score(y_true, y_pred, average='macro')
    micro = f1_score(y_true, y_pred, average='micro')
    acc = accuracy_score(y_true, y_pred)
    return macro, micro, acc

def get_f1_binary(file):
    json_file = json.load(open(filename))
    y_true = []
    y_pred = []
    for _,elm in json_file["results"].items():
        chosen = []
        for k, v in elm["PRED"].items(): 
            if v == "true":
                chosen.append(classes.index(k))
        if(len(chosen)==0): chosen = [0]
        y_pred.append(random.choice(chosen))
        y_true.append(classes.index(elm["GOLD"]))
    macro = f1_score(y_true, y_pred, average='macro')
    micro = f1_score(y_true, y_pred, average='micro')
    acc = accuracy_score(y_true, y_pred)
    return macro, micro, acc

acts = ["offerbooked","greet","reqmore","nobook","bye","offerbook","welcome","recommend","select","nooffer","book","inform","request"]

def get_f1_ACT(file):
    json_file = json.load(open(filename))
    y_true = []
    y_pred = []
    for _,elm in json_file["results"].items():
        pred = [0 for _ in acts]
        for k, v in elm["PRED"].items(): 
            if v == "true":
                pred[acts.index(k)] = 1
    
        y_pred.append(pred)
        y_true.append([1 if a in elm["GOLD"] else 0 for a in acts])
        # print(pred)
        # print([1 if a in elm["GOLD"] else 0 for a in acts])
        # break
    macro = f1_score(sum(y_true,[]), sum(y_pred,[]), average='macro')
    micro = f1_score(sum(y_true,[]), sum(y_pred,[]), average='micro')
    acc = accuracy_score(sum(y_true,[]), sum(y_pred,[]))
    return macro, micro, acc

# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BLEU metric implementation.
"""


def moses_multi_bleu(hypotheses, references, lowercase=False):
    """Calculate the bleu score for hypotheses and references
    using the MOSES ulti-bleu.perl script.
    Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script
    Returns:
    The BLEU score as a float32 value.
    """

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    multi_bleu_path = "utils/multi-bleu.perl"
    os.chmod(multi_bleu_path, 0o755)


    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()


     # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                print("multi-bleu.perl script returned non-zero exit code")
                print(error.output)
                bleu_score = np.float32(0.0)

    # Close temp files
    hypothesis_file.close()
    reference_file.close()
    return bleu_score

def get_BLEU(file):
    json_file = json.load(open(filename))
    PRED = []
    GOLD = []
    for _,elm in json_file["results"].items():
        PRED.append(elm["PRED"])
        GOLD.append(elm["GOLD"])
    BLEU = moses_multi_bleu(np.array(PRED),np.array(GOLD))
    return BLEU, [[p] for p in PRED]


if __name__ == "__main__":

    ## Slot filling
    domain_list = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
    model_list = ["gpt2_2", "gpt2_20","gpt2_30", "gpt2-large_2", "gpt2-large_20","gpt2-large_30", "gpt2-xl_2", "gpt2-xl_20","gpt2-xl_30"]
    table = []
    for model in model_list:
        temp = {"Model":model.split("_")[0],"Shots":model.split("_")[1]}
        avg = []
        for domain in domain_list:
            filename = os.path.join("results", f"{model}_{domain}.json")
            F1 = eval_f1(filename)
            temp[domain[-7:]] = F1
            avg.append(F1)
        temp["Avg"] = np.mean(avg)  
        table.append(temp)
    print(tabulate(table,headers="keys",tablefmt="latex",floatfmt=".4f"))
    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    color = sns.color_palette(flatui)
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    x = [1,10,15]
    y_s = [t["Avg"] for t in table if t["Model"] == "gpt2"]
    y_l = [t["Avg"] for t in table if t["Model"] == "gpt2-large"]
    y_xl = [t["Avg"] for t in table if t["Model"] == "gpt2-xl"]
    ax[0].plot(x, y_s,label="GPT2-117M",marker="s",color=color[-1])
    ax[0].plot(x, y_l,label="GPT2-762M",marker="^",color=color[-2])
    ax[0].plot(x, y_xl,label="GPT2-1.54B",marker="o",color=color[-3])
    ax[0].hlines(53.62,1,15,label="20-shots CT",linewidth=3,color=color[2],linestyles='dashed')
    ax[0].hlines(56.53,1,15,label="20-shots RZT",linewidth=3,color=color[1],linestyles='dashdot')
    ax[0].hlines(63.17,1,15,label="20-shots Coach",linewidth=3,color=color[0],linestyles='dotted')

    ax[0].set(xlabel='shots', ylabel='F1-SCORE',
        title='NLU: SLOT FILLING')
    ax[0].grid()
    ax[0].legend()


    # Intent Recognition
    model_list = ["gpt2_1_False","gpt2_5_False","gpt2_9_False",
                  "gpt2-large_1_False","gpt2-large_5_False","gpt2-large_9_False",
                  "gpt2-xl_1_False","gpt2-xl_5_False","gpt2-xl_9_False"
                ]
    table = []
    for model in model_list:
        filename = os.path.join("results", f"INTENT_{model}.json")
        macro, micro, acc = get_f1(filename)
        table.append({"Model":model.split("_")[0],"Shots":model.split("_")[1],"Mode":"direct","Micro":micro,"Macro":macro,"Acc":acc})

    table = []
    model_list = ["gpt2_1_True","gpt2_5_True","gpt2_10_True","gpt2_25_True",
                  "gpt2-large_1_True","gpt2-large_5_True","gpt2-large_10_True","gpt2-large_25_True",
                  "gpt2-xl_1_True","gpt2-xl_5_True","gpt2-xl_10_True","gpt2-xl_25_True"
                ]

    for model in model_list:
        filename = os.path.join("results", f"INTENT_{model}.json")
        macro, micro, acc = get_f1_binary(filename)
        table.append({"Model":model.split("_")[0],"Shots":model.split("_")[1],"Mode":"binary","Micro":micro,"Macro":macro,"Acc":acc*100})
    print(tabulate(table,headers="keys",tablefmt="latex",floatfmt=".4f"))
    # fig, ax = plt.subplots(figsize=(7,5))
    x = [1,2,5,10]
    y_s = [t["Acc"] for t in table if t["Model"] == "gpt2"]
    y_l = [t["Acc"] for t in table if t["Model"] == "gpt2-large"]
    y_xl = [t["Acc"] for t in table if t["Model"] == "gpt2-xl"]
    ax[1].plot(x, y_s,label="GPT2-117M",marker="s",color=color[-1])
    ax[1].plot(x, y_l,label="GPT2-762M",marker="^",color=color[-2])
    ax[1].plot(x, y_xl,label="GPT2-1.54B",marker="o",color=color[-3])
    ax[1].hlines(82.46,1,10,label="10-shots RoBERTa",linewidth=3,color=color[2],linestyles='dashed')

    ax[1].set(xlabel='shots', ylabel='ACC',
        title='NLU: INTENT CLASSIFICATION')
    ax[1].grid()
    ax[1].legend()

    fig.savefig("results/NLU.png")

    #NLG
    table = [
        {"Model":"SC-LSTM","restaurant":15.90,"laptop":21.98,"hotel":31.30,"tv":22.39,"attraction":7.76,"train": 6.08,"taxi":11.61,"Avg":0.0},
        {"Model":"GPT-2","restaurant":29.48,"laptop":27.43,"hotel":35.75,"tv":28.47,"attraction":16.11,"train":13.72,"taxi":16.27,"Avg":0.0},
        {"Model":"SC-GPT","restaurant":38.08,"laptop":32.73,"hotel":38.25,"tv":32.95,"attraction":20.69,"train":17.21,"taxi":19.70,"Avg":0.0},
    ]
                    
    tableERR = [
        {"Model":"SC-LSTM","restaurant":48.02,"laptop":80.48,"hotel":31.54,"tv":64.62,"attraction":367.12,"train":189.88,"taxi":61.45,"Avg":0.0},
        {"Model":"GPT-2","restaurant":13.47,"laptop":11.26,"hotel":11.54,"tv":9.44,"attraction":21.10,"train":19.26,"taxi":9.52,"Avg":0.0},
        {"Model":"SC-GPT","restaurant":3.89,"laptop":3.39,"hotel":2.75,"tv":3.38,"attraction":12.72,"train":7.74,"taxi":3.57,"Avg":0.0},
    ]
    domains = ["attraction","hotel","restaurant","taxi","train","laptop","tv"]
    model_list = ["gpt2_5","gpt2_10","gpt2_20","gpt2-large_5","gpt2-large_10","gpt2-large_20","gpt2-xl_5","gpt2-xl_10","gpt2-xl_20"]
    for model in model_list:
        temp = {"Model":model}
        temp_E = {"Model":model}
        avg_B = []
        avg_E = []
        for d in domains:
            # print(d)
            filename = os.path.join("results", f"RNNLG_{d}_{model}.json")
            B_O,pred = get_BLEU(filename)
            if(d in ["attraction","taxi","train"]):
                ERR = 0.0
                temp_E[d] = ERR
                temp[d] = B_O #scorer.score(pred,f'models/SC_GPT/test_results/{d}.test.txt')
                avg_B.append(B_O)
                avg_E.append(ERR)
            else:
                BLEU,ERR = score(d,pred)
                temp[d] = BLEU*100
                temp_E[d] = ERR
                avg_B.append(BLEU*100)
                avg_E.append(ERR)
        temp["Avg"] = np.mean(avg_B)
        temp_E["Avg"] = np.mean(avg_E)
        table.append(temp)
        tableERR.append(temp_E)
    print("BLEU")
    print(tabulate(table,headers="keys",tablefmt="latex",floatfmt=".2f"))
    print()
    print("ERR")
    print(tabulate(tableERR,headers="keys",tablefmt="latex",floatfmt=".2f"))

    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    color = sns.color_palette(flatui)
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    x = [5,10,20]
    y_s = [t["Avg"] for t in table if t["Model"].split("_")[0] == "gpt2"]
    y_l = [t["Avg"] for t in table if t["Model"].split("_")[0] == "gpt2-large"]
    y_xl = [t["Avg"] for t in table if t["Model"].split("_")[0] == "gpt2-xl"]
    ax[0].plot(x, y_s,label="GPT2-117M",marker="s",color=color[-1])
    ax[0].plot(x, y_l,label="GPT2-762M",marker="^",color=color[-2])
    ax[0].plot(x, y_xl,label="GPT2-1.54B",marker="o",color=color[-3])
    ax[0].hlines(16.717,5,20,linewidth=3,color=color[2],linestyles='dashed',label="50-shots SC-LSTM")
    ax[0].hlines(23.89,5,20,linewidth=3,color=color[1],linestyles='dashdot',label="50-shots GPT2")
    ax[0].hlines(28.51,5,20,linewidth=3,color=color[0],linestyles='dotted',label="50-shots SC-GPT2")

    ax[0].set(xlabel='shots', ylabel='BLEU',
        title='NLG: BLEU')
    ax[0].grid()
    ax[0].legend()

    y_s = [t["Avg"] for t in tableERR if t["Model"].split("_")[0] == "gpt2"]
    y_l = [t["Avg"] for t in tableERR if t["Model"].split("_")[0] == "gpt2-large"]
    y_xl = [t["Avg"] for t in tableERR if t["Model"].split("_")[0] == "gpt2-xl"]
    ax[1].plot(x, y_s,label="GPT2-117M",marker="s",color=color[-1])
    ax[1].plot(x, y_l,label="GPT2-762M",marker="s",color=color[-2])
    ax[1].plot(x, y_xl,label="GPT2-1.54B",marker="s",color=color[-3])
    ax[1].hlines(120.44428571429,5,20,linewidth=3,color=color[2],linestyles='dashed',label="50-shots SC-LSTM")
    ax[1].hlines(13.6557,5,20,linewidth=3,color=color[1],linestyles='dashdot',label="50-shots GPT2")
    ax[1].hlines(5.34857,5,20,linewidth=3,color=color[0],linestyles='dotted',label="50-shots SC-GPT2")

    ax[1].set(xlabel='shots', ylabel='ERR',
        title='NLG: SLOT ERROR RATE')
    ax[1].grid()
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("results/NLG.png")






    ## Speech ACT Detection
    model_list = ["gpt2_10","gpt2_20","gpt2_29",
                  "gpt2-large_10","gpt2-large_20","gpt2-large_29",
                  "gpt2-xl_10","gpt2-xl_20","gpt2-xl_29"
                ]
    table = []
    for model in model_list:
        filename = os.path.join("results", f"ACT_{model}.json")
        macro, micro, acc = get_f1_ACT(filename)
        # print(filename,macro, micro, acc)
        table.append({"Model":model.split("_")[0],"Shots":model.split("_")[1],"Micro":micro*100,"Macro":macro*100,"Acc":acc})
    print(tabulate(table,headers="keys",tablefmt="latex",floatfmt=".4f"))

    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    color = sns.color_palette(flatui)
    fig, ax = plt.subplots()
    x = [5,10,15]
    y_s = [t["Micro"] for t in table if t["Model"] == "gpt2"]
    y_l = [t["Micro"] for t in table if t["Model"] == "gpt2-large"]
    y_xl = [t["Micro"] for t in table if t["Model"] == "gpt2-xl"]
    ax.plot(x, y_s,label="GPT2-117M",marker="s",color=color[-1])
    ax.plot(x, y_l,label="GPT2-762M",marker="s",color=color[-2])
    ax.plot(x, y_xl,label="GPT2-1.54B",marker="s",color=color[-3])
    ax.hlines(84.0,5,15,label="500-shots BERT",linewidth=3,color=color[2],linestyles='dashed')
    ax.hlines(87.5,5,15,label="500-shots ToD-BERT",linewidth=3,color=color[1],linestyles='dashdot')


    ax.set(xlabel='shots', ylabel='F1',
        title='DM: SPEECH ACT DETECTION')
    ax.grid()
    ax.legend()
    fig.savefig("results/ACT.png")









    # DST RESULTS PRED DST
    model_list = ["gpt2_10","gpt2_20","gpt2_30",
                  "gpt2-large_10","gpt2-large_20","gpt2-large_30",
                  "gpt2-xl_10","gpt2-xl_20","gpt2-xl_30"
                ]

    table = []
    for model in model_list:
        filename = os.path.join("results", f"DST_{model}.json")
        json_file = json.load(open(filename))
        json_test = json.load(open(".data/MWoZ_2.1/DST/DST_TEST.json"))

        joint_acc = []
        slot_acc = []
        for (gold_OBJ),(_,elm) in zip(json_test,json_file["results"].items()):
            assert elm["id"] == gold_OBJ['id']
            assert elm["turn_id"] == gold_OBJ['turn_id']
            gold_DST = {s:v for s,v in zip(gold_OBJ["slots"],gold_OBJ["slot_values"])}
            if gold_OBJ["turn_id"] == 0:
                pred_DST = {s:"none" for s,v in gold_DST.items()}
            for s,v in elm["PRED"].items():
                pred_DST[elm["domain"]+"-"+s] = v
            flag = True
            for (s_g,v_g), (s_p,v_p) in zip(gold_DST.items(),pred_DST.items()):
                assert s_g==s_p
                if(v_g == v_p): slot_acc.append(1)
                else: 
                    slot_acc.append(0)
                    flag = False
            if(flag): joint_acc.append(1)
            else: joint_acc.append(0)

        table.append({"Model":model.split("_")[0],"Shots":model.split("_")[1],"Joint":np.mean(joint_acc)*100,"Slot":np.mean(slot_acc)*100})
    
    print(tabulate(table,headers="keys",tablefmt="latex",floatfmt=".1f"))

    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    color = sns.color_palette(flatui)
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    x = [5,10,15]
    y_s = [t["Joint"] for t in table if t["Model"] == "gpt2"]
    y_l = [t["Joint"] for t in table if t["Model"] == "gpt2-large"]
    y_xl = [t["Joint"] for t in table if t["Model"] == "gpt2-xl"]
    ax[0].plot(x, y_s,label="GPT2-117M",marker="s",color=color[-1])
    ax[0].plot(x, y_l,label="GPT2-762M",marker="s",color=color[-2])
    ax[0].plot(x, y_xl,label="GPT2-1.54B",marker="s",color=color[-3])
    ax[0].hlines(7.6,5,15,label="500-shots BERT",linewidth=3,color=color[2],linestyles='dashed')
    ax[0].hlines(10.3,5,15,label="500-shots ToD-BERT",linewidth=3,color=color[1],linestyles='dashdot')


    ax[0].set(xlabel='shots', ylabel='Joint',
        title='DST: Joint')
    ax[0].grid()
    ax[0].legend()

    y_s = [t["Slot"] for t in table if t["Model"] == "gpt2"]
    y_l = [t["Slot"] for t in table if t["Model"] == "gpt2-large"]
    y_xl = [t["Slot"] for t in table if t["Model"] == "gpt2-xl"]
    ax[1].plot(x, y_s,label="GPT2-117M",marker="s",color=color[-1])
    ax[1].plot(x, y_l,label="GPT2-762M",marker="s",color=color[-2])
    ax[1].plot(x, y_xl,label="GPT2-1.54B",marker="s",color=color[-3])
    ax[1].hlines(84.1,5,15,label="500-shots BERT",linewidth=3,color=color[2],linestyles='dashed')
    ax[1].hlines(86.7,5,15,label="500-shots ToD-BERT",linewidth=3,color=color[1],linestyles='dashdot')

    ax[1].set(xlabel='shots', ylabel='Slot',
        title='DST: Slot')
    ax[1].grid()
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("results/DST.png")