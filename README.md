# ShipTrajPredict
多船舶轨迹预测：根据历史的轨迹来预测未来时刻对最后一个已知时间点的经纬度偏移量，最后还原成经纬度

### 所有代码均放在 TrajPredictapi 文件夹内
#### TrajPredictapi为一个 fastapi 项目，启动所需要的库请看 requirements.txt 或 environment.yml

#### 目前 项目中仅有一个算法：简易的 transformer 模型

#### 数据集来源：1、丹麦海事局： http://aisdata.ais.dk/?prefix=
               2、美国NOAA：https://hub.marinecadastre.gov/pages/vesseltraffic
