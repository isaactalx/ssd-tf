# ssd-tf
参考https://github.com/balancap/SSD-Tensorflow 的一个简化版SSD300实现

#### 使用：<br>
#### 推理：<br>
1.解压./checkpoints/ssd_300_vgg.ckpt.zip<br>
2.python ./inference.py --image {file_path}<br>

#### 训练：<br>
1.python voc_to_tfrecords.py生成tfrecord文件
2.python ./train.py --batch_size {batch_size} --max_steps {max_steps}<br>
