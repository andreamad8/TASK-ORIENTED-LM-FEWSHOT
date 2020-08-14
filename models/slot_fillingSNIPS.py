from utils.load_snips import SNIPS
from collections import defaultdict
import json
from tqdm import tqdm

def predict(args,tokenizer,model,train, test, slots):
    base_prefixes = defaultdict(str)
    for slot in slots:
        for _, b in enumerate(train[slot]):
            base_prefixes[slot] += f"{b[0]}=>{slot}={b[1]}\n"
    
    total_results = {}
    for idx_b, b in tqdm(enumerate(test),total=len(test)):
        result = {}
        sequence_slot = {}
        for slot, base_prefix in base_prefixes.items():
            # print("SLOT:",slot)
            prefix = base_prefix+f"{b[0]}=>{slot}="

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
            result[slot] = rep[:rep.find("\n")] ## cleaning 
            sequence_slot[slot] = total_sequence

        total_results[idx_b] = {"PRED":result, "query":sequence_slot, "text":b[0],"TAGS":b[1],"GOLD":b[2]}
        # if(idx_b==10):break
    return total_results

def SNIPS_slot(args,tokenizer,model):
    shot = args.shots
    domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather", 
                  "PlayMusic", "RateBook", "SearchCreativeWork", 
                  "SearchScreeningEvent"]
    for d in domain_set:
        print(f"Domain:{d}")
        train, test, slots = SNIPS(args,domain=d,shots=shot)
        results = predict(args,tokenizer,model,train, test, slots)
        results = {"meta":str(args), "results":results}
        with open(f"results/{args.model_name_or_path}_{args.shots}_{d}_{args.balanced}.json", "w", encoding="utf-8") as f:
            json.dump(results,f,indent=4)