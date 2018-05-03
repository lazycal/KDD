cd ..
/Users/lazycal/anaconda/bin/python -u DataMatching.py > DataMatching.log 2>&1
cd ./ljk
/Users/lazycal/anaconda/bin/python -u train.py output/ans_l12 --outcsv ./ans_l12.csv --lag 12 --strategy Strategy1 --deploy > ans_l12.log 2>&1
/Users/lazycal/anaconda/bin/python -u train.py output/ans_l24 --outcsv ./ans_l24.csv --lag 12 --strategy Strategy2 --deploy > ans_l24.log 2>&1
/Users/lazycal/anaconda/bin/python -u api_submit.py > api_submit.log 2>&1
