#! /bin/bash
unzip -d ./data/Weibo ./data/Weibo/weibotree.txt.zip
pip install -U torch==1.4.0 numpy==1.18.1
pip install -r requirements.txt
#Generate graph data and store in /data/Weibograph
python ./Process/getWeibograph.py
#Generate graph data and store in /data/Twitter15graph
python ./Process/getTwittergraph.py Twitter15
#Generate graph data and store in /data/Twitter16graph
python ./Process/getTwittergraph.py Twitter16
#Reproduce the experimental results.
#python ./model/Weibo/BiGCN_Weibo.py 100
python ./model/Weibo/BiGCN_mine.py 1
python ./model/Twitter/BiGCN_Twitter.py Twitter15 100
#python ./model/Twitter/BiGCN_Twitter.py Twitter16 100

# To run the BiGCN_BERT model.
python ./model/Weibo/BiGCN_BERT.py 1
# To run the GAT_BERT model.
python ./model/Weibo/GAT_BERT.py
