CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2 --model_name_or_path gpt2 --shots 1 --task SNIPS_INTENT --binary
CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2 --model_name_or_path gpt2 --shots 5 --task SNIPS_INTENT --binary
CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2 --model_name_or_path gpt2 --shots 10 --task SNIPS_INTENT --binary
CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2 --model_name_or_path gpt2 --shots 25 --task SNIPS_INTENT --binary

CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2-large --model_name_or_path gpt2-large --shots 1 --task SNIPS_INTENT --binary
CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2-large --model_name_or_path gpt2-large --shots 5 --task SNIPS_INTENT --binary
CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2-large --model_name_or_path gpt2-large --shots 10 --task SNIPS_INTENT --binary
CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2-large --model_name_or_path gpt2-large --shots 25 --task SNIPS_INTENT --binary

CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2-xl --model_name_or_path gpt2-xl --shots 1 --task SNIPS_INTENT --binary
CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2-xl --model_name_or_path gpt2-xl --shots 5 --task SNIPS_INTENT --binary
CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2-xl --model_name_or_path gpt2-xl --shots 10 --task SNIPS_INTENT --binary
CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2-xl --model_name_or_path gpt2-xl --shots 25 --task SNIPS_INTENT --binary

