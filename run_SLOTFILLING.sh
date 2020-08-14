CUDA_VISIBLE_DEVICES=5 python main.py --model_type=gpt2 --model_name_or_path gpt2 --shots 2
CUDA_VISIBLE_DEVICES=2 python main.py --model_type=gpt2 --model_name_or_path gpt2 --shots 20
CUDA_VISIBLE_DEVICES=2 python main.py --model_type=gpt2 --model_name_or_path gpt2 --shots 30

CUDA_VISIBLE_DEVICES=4 python main.py --model_type=gpt2-large --model_name_or_path gpt2-large --shots 2
CUDA_VISIBLE_DEVICES=2 python main.py --model_type=gpt2-large --model_name_or_path gpt2-large --shots 20
CUDA_VISIBLE_DEVICES=2 python main.py --model_type=gpt2-large --model_name_or_path gpt2-large --shots 30

CUDA_VISIBLE_DEVICES=5 python main.py --model_type=gpt2-xl --model_name_or_path gpt2-xl --shots 2
CUDA_VISIBLE_DEVICES=3 python main.py --model_type=gpt2-xl --model_name_or_path gpt2-xl --shots 20
CUDA_VISIBLE_DEVICES=3 python main.py --model_type=gpt2-xl --model_name_or_path gpt2-xl --shots 30