from utils.load_MWOZ import MWOZ_getACT
from collections import defaultdict
import json
from tqdm import tqdm
import numpy as np
import random
random.seed(0)

def predict_binary(args,tokenizer,model,train, test):
    base_prefixes = {}
    for domain in train.keys():
        temp = defaultdict(str)
        for act, li in train[domain].items():
            random.shuffle(li)
            for b in li:
                temp[act] += f"{b[0]}=>{act}={b[1]}\n"
            # print(f"{domain} {act} {len(tokenizer.encode(temp[act],add_special_tokens=False))}")
        base_prefixes[domain] = temp
    
    total_results = {}
    acc = []
    for idx_b, b in tqdm(enumerate(test),total=len(test)):
        result = {}
        sequence_intent = {}
        # print(base_prefixes[b["domain"]])
        for act, base_prefix in base_prefixes[b["domain"]].items():
            # print("SLOT:",slot)
            prefix = base_prefix+f"{b['sys']}=>{act}="
            encoded_prompt = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt")
            
            input_ids = encoded_prompt.to(args.device)


            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=args.length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
            )

            # Remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            generated_sequences = []
            generated_answers = []
            for _, generated_sequence in enumerate(output_sequences):
                # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                # Remove all text after the stop token
                # text = text[: text.find(args.stop_token) if args.stop_token else None]
                generated_answers.append(text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :])
                # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
                total_sequence = (
                    prefix + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
                )

                generated_sequences.append(total_sequence)

            rep = generated_answers[0].lower()
            result[act] = rep[:rep.find("\n")] ## cleaning 
            sequence_intent[act] = total_sequence

        total_results[idx_b] = {"domain":b["domain"],"PRED":result, "query":sequence_intent, "text":b['sys'],"GOLD":b['speech_act']}
        # with open("temp.json", "w", encoding="utf-8") as f:
        #     json.dump(total_results,f,indent=4)

        # if(idx_b==100):break
    return total_results


def MWOZ_ACT(args,tokenizer,model):
    train, test = MWOZ_getACT(args)

    results = predict_binary(args,tokenizer,model,train, test)

    results = {"meta":str(args), "results":results}
    with open(f"results/ACT_{args.model_name_or_path}_{args.shots}.json", "w", encoding="utf-8") as f:
        json.dump(results,f,indent=4)

            