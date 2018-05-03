# python -u train.py output/1-12-tune400-max_bin511 --lag 12 --strategy Strategy1 --es -1 --max-iter 400 --max-bin 511 > 1-12-tune400-max_bin511.log
python -u train.py output/1-24-tune400-max_bin511 --lag 24 --strategy Strategy1 --es -1 --max-iter 400 --max-bin 511 > 1-24-tune400-max_bin511.log
python -u train.py output/1-24-tune400 --lag 24 --strategy Strategy1 --es -1 --max-iter 400 > 1-24-tune400.log
# python -u train.py output/1-48 --lag 48 --strategy Strategy1 > 1-48.log
