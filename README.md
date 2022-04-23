# e-commerce-ad-rec-sys
## 电商广告推荐系统
该项目数据基于阿里巴巴提供的一个淘宝展示广告点击率预估数据集Ali_Display_Ad_Click.  
数据集下载地址：https://tianchi.aliyun.com/dataset/dataDetail?dataId=56  
### 数据存储
1.先在本地创建一个dataset文件夹，将数据下载到该文件夹下，打开terminal命令行终端，执行如下命令将数据存放到hdfs下：
```
# 先在hdfs创建该项目文件夹recsys，再将数据存放到该目录下
hadoop fs -mkdir /recsys
hadoop fs -put ./dataset /recsys
# 查看是否成功存储
hadoop fs -ls /recsys/dataset
```
