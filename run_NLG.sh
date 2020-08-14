CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2 --model_name_or_path gpt2 --shots 5  --task RNNLG --length 50
CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2 --model_name_or_path gpt2 --shots 10  --task RNNLG --length 50
CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2 --model_name_or_path gpt2 --shots 20  --task RNNLG --length 50


CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2-large --model_name_or_path gpt2-large --shots 5  --task RNNLG --length 50
CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2-large --model_name_or_path gpt2-large --shots 10  --task RNNLG --length 50
CUDA_VISIBLE_DEVICES=1 python main.py --model_type=gpt2-large --model_name_or_path gpt2-large --shots 20  --task RNNLG --length 50

CUDA_VISIBLE_DEVICES=3 python main.py --model_type=gpt2-xl --model_name_or_path gpt2-xl --shots 5  --task RNNLG --length 50
CUDA_VISIBLE_DEVICES=3 python main.py --model_type=gpt2-xl --model_name_or_path gpt2-xl --shots 10  --task RNNLG --length 50
CUDA_VISIBLE_DEVICES=3 python main.py --model_type=gpt2-xl --model_name_or_path gpt2-xl --shots 20  --task RNNLG --length 50