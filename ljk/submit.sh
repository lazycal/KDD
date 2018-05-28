cd ..
/Users/lazycal/anaconda/bin/python -u DataMatching.py > DataMatching.log 2>&1
cp ./data/data.csv ./ljk/subs/
cd ./ljk
/Users/lazycal/anaconda/bin/python -u train.py output/ans_l12     --outcsv ./ans_l12.csv     --lag 12 --strategy Strategy18 --deploy > ans_l12.log     2>&1
# /Users/lazycal/anaconda/bin/python -u train-deploy.py output/ans_l12_log --outcsv ./ans_l12_log.csv --lag 12 --strategy Strategy5 --deploy > ans_l12_log.log 2>&1
# /Users/lazycal/anaconda/bin/python merger.py ans_l12_log.csv ./score-5-12.csv ans_l12.csv ./score-1-12.csv ans-better.csv
# /Users/lazycal/anaconda/bin/python -u silam/genapisubmit.py `date +"%Y-%m-%d"` > genapisubmit.log 2>&1
# /Users/lazycal/anaconda/bin/python -u silam/merger.py > merger.log 2>&1
/Users/lazycal/anaconda/bin/python -u api_submit.py > api_submit.log 2>&1
time=`date +"%m_%d"`
# mv merger.log subs/merger_$time.log
# mv genapisubmit.log subs/genapisubmit_$time.log
# mv silam.csv subs/silam_$time.csv
mv ans_l12.log subs/ans_l12_$time.log
mv ans_l12.csv subs/ans_l12_$time.csv
mv output/ans_l12 subs/ans_l12_$time
mv ../DataMatching.log ../DataMatching_$time.log
mv ../data/data.csv ../data/data_$time.csv