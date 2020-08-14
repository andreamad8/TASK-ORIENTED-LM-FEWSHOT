from utils.load_snips import SNIPS_get_intent
from collections import defaultdict
import json
from tqdm import tqdm
import numpy as np
import random

def predict(args,tokenizer,model,train, test):
    base_prefix = ""
    for _, b in enumerate(train):
        base_prefix += f"{b[0]}=>{b[1]}\n"

    total_results = {}
    acc = []
    for idx_b, b in tqdm(enumerate(test),total=len(test)):
        # print("SLOT:",slot)
        prefix = base_prefix+f"{b[0]}=>"

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
        rep = rep[:rep.find("\n")]
        if(rep in b[1] or b[1] in rep): acc.append(1)
        else: acc.append(0)
        total_results[idx_b] = {"PRED":rep, "query":total_sequence, "text":b[0],"GOLD":b[1]}
        # print(np.mean(acc))
        # if(idx_b==100):break
    return total_results

def predict_binary(args,tokenizer,model,train, test):
    base_prefixes = defaultdict(str)
    for intent in train.keys():
        for _, b in enumerate(train[intent]):
            base_prefixes[intent] += f"{b[0]}=>{intent}={b[1]}\n"
    
    total_results = {}
    acc = []
    for idx_b, b in tqdm(enumerate(test),total=len(test)):
        result = {}
        sequence_intent = {}
        for intent, base_prefix in base_prefixes.items():
            # print("SLOT:",slot)
            prefix = base_prefix+f"{b[0]}=>{intent}="

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
            result[intent] = rep[:rep.find("\n")] ## cleaning 
            sequence_intent[intent] = total_sequence
        # chosen = []
        # for k, v in result.items(): 
        #     if v == "true":
        #         chosen.append(k)
        # if(random.choice(chosen) == b[1]): acc.append(1)
        # else: acc.append(0)
        # print(np.mean(acc))
        total_results[idx_b] = {"PRED":result, "query":sequence_intent, "text":b[0],"GOLD":b[1]}
        # with open("temp.json", "w", encoding="utf-8") as f:
        #     json.dump(total_results,f,indent=4)

        # if(idx_b==100):break
    return total_results


def SNIPS_intent(args,tokenizer,model):
    train, valid, test = SNIPS_get_intent(args)
    if(args.binary): 
        results = predict_binary(args,tokenizer,model,train, test)
    else:
        results = predict(args,tokenizer,model,train, test)
    results = {"meta":str(args), "results":results}
    with open(f"results/INTENT_{args.model_name_or_path}_{args.shots}_{args.binary}.json", "w", encoding="utf-8") as f:
        json.dump(results,f,indent=4)

            