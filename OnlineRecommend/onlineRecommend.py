import redis
import json
import pandas as pd
from pyspark.ml.linalg import DenseVector
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel


class OnlineRecommend(object):
    """基于用户召回的候选物品进行LR逻辑回归排序，为用户推荐top-n个物品"""
    SPARK_APP_NAME = "OnlineRecommended"
    SPARK_URL = "spark://ymh:7077"
    MODEL_PATH = "hdfs://localhost:9000/recsys/models/CTRModel_Normal.obj"

    def initial(self):
        """初始化推荐系统"""
        conf = self.get_spark_conf()
        self.spark = self.get_or_create_spark(conf)

        self.set_rela()
        self.create_redis_client()
        self.load_model()

    def get_spark_conf(self):
        """获取建立spark连接的配置参数"""
        conf = SparkConf()
        config = [
            ("spark.app.name", self.SPARK_APP_NAME),
            ("spark.executor.memory", "10g"),
            ("spark.master", self.SPARK_URL),
            ("spark.executor.cores", "3")
        ]
        conf.setAll(config)
        return conf

    def get_or_create_spark(self, conf=None):
        """利用配置好的conf参数对象，创建spark session会话"""
        spark = SparkSession.builder.config(conf=conf).getOrCreate()
        return spark

    def set_rela(self):
        """设置类别特征转换为onehot编码对应的关系"""
        self.pvalue_level_rela = {-1: 0, 3: 3, 1: 2, 2: 1}
        self.new_user_class_level_rela = {-1: 0, 3: 2, 1: 4, 4: 3, 2: 1}
        self.pid_rela = {"430548_1007": 0, "430549_1007": 1}

    def create_redis_client(self):
        """建立redis客户端连接"""
        self.client_of_recall = redis.StrictRedis(host="localhost", port=6379, password="123456", db=9)
        self.client_of_features = redis.StrictRedis(host="localhost", port=6379, password="123456", db=10)

    def create_dataset(self, user_id, pid):
        """创建LR排序模型的输入数据集"""
        # 从redis缓存中获取用户特征和召回物品
        user_feature = json.loads(self.client_of_features.hget("user_features", user_id).decode())
        recall_sets = self.client_of_recall.smembers(user_id)
        result = []

        # 遍历召回物品集合
        for adgroupId in recall_sets:
            # 获取广告特征
            adgroupId = int(adgroupId)
            ad_feature = json.loads(self.client_of_features.hget("ad_features", adgroupId).decode())
            # 合并用户和广告特征
            features = {}
            features.update(user_feature)
            features.update(ad_feature)
            for k, v in features.items():
                if v is None:
                    features[k] = -1

            features_col = ["price", "cms_segid", "cms_group_id", "final_gender_code", "age_level", "shopping_level",
                            "occupation", "pid", "pvalue_level", "new_user_class_level"]
            price = float(features["price"])
            cms = features["cms_segid"]
            cms_group_id = features["cms_group_id"]
            final_gender_code = features["final_gender_code"]
            age_level = features["age_level"]
            shopping_level = features["shopping_level"]
            occupation = features["occupation"]

            pid_value = [0 for i in range(2)]
            pvalue_level_value = [0 for i in range(4)]
            new_user_class_level_value = [0 for i in range(5)]
            pid_value[self.pid_rela[pid]] = 1
            pvalue_level_value[self.pvalue_level_rela[int(features["pvalue_level"])]] = 1
            new_user_class_level_value[self.new_user_class_level_rela[int(features["new_user_class_level"])]] = 1

            vector = DenseVector(pid_value + [price, cms, cms_group_id, final_gender_code, age_level, shopping_level,
                                 occupation] + pvalue_level_value + new_user_class_level_value)

            result.append((user_id, adgroupId, vector))
        return result

    def load_model(self):
        """加载训练好的模型"""
        self.CTR_model = LogisticRegressionModel.load(self.MODEL_PATH)

    def handle_request(self, user_id, pid, n=10):
        """处理用户的请求，返回推荐物品"""
        if pid not in ["430548_1007", "430549_1007"]:
            raise Exception("Invalid pid value! It should be one of the: 430548_1007, 430549_1007")

        pdf = pd.DataFrame(self.create_dataset(user_id, pid), columns=["userId", "adgroupId", "features"])
        dataset = self.spark.createDataFrame(pdf)
        prediction = self.CTR_model.transform(dataset).sort("probability")
        return [i.adgroupId for i in prediction.select("adgroupId").head(n)]


if __name__ == "__main__":
    print("推荐系统正在启动...")
    obj = OnlineRecommend()
    obj.initial()
    print("推荐系统启动成功.")
    while True:
        user_id = input("请输入用户ID：")
        _pid = input("请输入广告资源位（输入1（默认）：430548_1007 | 输入2：430549_1007）：")
        if _pid.strip() == "1":
            pid = "430548_1007"
        elif _pid.strip() == "2":
            pid = "430549_1007"
        else:
            pid = "430548_1007"

        n = input("请输入需要推荐的广告个数（默认10，最大500）：")
        if n.strip() == "":
            n = 10
        if int(n) > 500:
            n = 500

        ret = obj.handle_request(int(user_id), pid, int(n))
        print("给用户%s推荐的广告ID列表为：" % user_id, ret)

        _ = input("继续请输入1，否则输入其他任意键退出：")
        if _.strip() != "1":
            break
