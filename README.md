# 心得：**WGAN网络的创建**

## Theory
* GAN-Loss
<img src="https://github.com/xiaoxiaokaiyan/New_Tensorflow_AE_VAE_FashionMnist_GAN_WGAN_Anime/blob/master/theory/GAN%20loss.PNG" width = 100% height =50% div align=left />

* WGAN-Gradient-Penalty
<img src="https://github.com/xiaoxiaokaiyan/New_Tensorflow_AE_VAE_FashionMnist_GAN_WGAN_Anime/blob/master/theory/WGAN-Gradient%20Penalty.PNG" width = 100% height =50% div align=left />

&nbsp;
<br/>


## Dependencies:
* &gt; GeForce GTX 1660TI
* Windows10
* python==3.6.12
* torch==1.0.0
* GPU环境安装包，下载地址：https://pan.baidu.com/s/14Oisbo9cZpP7INQ6T-3vwA 提取码：z4pl （网上找的）
```
  Anaconda3-5.2.0-Windows-x86_64.exe
  cuda_10.0.130_411.31_win10.exe
  cudnn-10.0-windows10-x64-v7.4.2.24.zip
  h5py-2.8.0rc1-cp36-cp36m-win_amd64.whl
  numpy-1.16.4-cp36-cp36m-win_amd64.whl
  tensorflow_gpu-1.13.1-cp36-cp36m-win_amd64.whl
  torch-1.1.0-cp36-cp36m-win_amd64.whl
  torchvision-0.3.0-cp36-cp36m-win_amd64.whl
```
<br/>


## Visualization Results
* AE生成结果对比
<img src="https://github.com/xiaoxiaokaiyan/New_Tensorflow_AE_VAE_GAN_FashionMnist/blob/master/result/AE%E7%94%9F%E6%88%90%E7%BB%93%E6%9E%9C%E5%AF%B9%E6%AF%94%E5%9B%BE%E7%89%87.png" width = 50% height =50%  div align=center />

* VAE随机生成第1代
<img src="https://github.com/xiaoxiaokaiyan/New_Tensorflow_AE_VAE_GAN_FashionMnist/blob/master/result/VAE%E9%9A%8F%E6%9C%BA%E7%94%9F%E6%88%90%E7%AC%AC1%E4%BB%A3%E5%9B%BE%E7%89%87.png" width = 50% height =50%  div align=center />


* VAE随机生成第9代
<img src="https://github.com/xiaoxiaokaiyan/New_Tensorflow_AE_VAE_GAN_FashionMnist/blob/master/result/VAE%E9%9A%8F%E6%9C%BA%E7%94%9F%E6%88%90%E7%AC%AC9%E4%BB%A3%E5%9B%BE%E7%89%87.png" width = 50% height =50% div align=center />

* WGAN生成第19700代（19800代开始，g-loss从1渐渐变成4，且稳定在4，生成的图片模糊，这个问题未解决）
<img src="https://github.com/xiaoxiaokaiyan/New_Tensorflow_AE_VAE_FashionMnist_GAN_WGAN_Anime/blob/master/result/wgan-19700.png" width = 50% height =50% div align=center />
&nbsp;
<br/>


## Public Datasets:
* fashion_mnist，是一个替代MNIST手写数字集的图像数据集。它是由Zalando（一家德国的时尚科技公司）旗下的研究部门提供。其涵盖了来自10种类别的共7万个不同商品的正面图片。Fashion-MNIST的大小、格式和训练集/测试集划分与原始的MNIST完全一致。60000/10000的训练测试数据划分，28x28的灰度图片。你可以直接用它来测试你的机器学习和深度学习算法性能，且不需要改动任何的代码。
* the Anime dataset should be prepared by yourself in ./data/faces/*.jpg,63565个彩色图片。
  * dataset link: [https://www.kaggle.com/splcher/animefacedataset](https://www.kaggle.com/splcher/animefacedataset)
<br/>

## Experience：
### （1）代码问题
```
      # [b, 28, 28] => [b, 28, 28]
      x_concat1 = tf.concat([x, x_hat], axis=0)

      # [b, 28, 28] => [2b, 28, 28]
      x_concat1 = tf.reshape(tf.concat([x, x_hat], axis=0),[-1, 28, 28])  ---------此处必须重新reshape，才能得到[2b, 28, 28]，才能生成Visualization Results第一幅图
```   

### （2）关于VAE和GAN的区别
  * 1.VAE和GAN都是目前来看效果比较好的生成模型，本质区别我觉得这是两种不同的角度，VAE希望通过一种显式(explicit)的方法找到一个概率密度，并通过最小化对数似函数的下限来得到最优解；
GAN则是对抗的方式来寻找一种平衡，不需要认为给定一个显式的概率密度函数。（李飞飞）
  * 2.简单来说，GAN和VAE都属于深度生成模型（deep generative models，DGM）而且属于implicit DGM。他们都能够从具有简单分布的随机噪声中生成具有复杂分布的数据（逼近真实数据分布），而两者的本质区别是从不同的视角来看待数据生成的过程，从而构建了不同的loss function作为衡量生成数据好坏的metric度量。
  * 3.要求得一个生成模型使其生成数据的分布 能够最小化与真实数据分布之间的某种分布差异度量，例如KL散度、JS散度、Wasserstein距离等。采用不同的差异度量会导出不同的loss function，比如KL散度会导出极大似然估计，JS散度会产生最原始GAN里的判别器，Wasserstein距离通过dual form会引入critic。而不同的深度生成模型，具体到GAN、VAE还是flow model，最本质的区别就是从不同的视角来看待数据生成的过程，从而采用不同的数据分布模型来表达。 [https://www.zhihu.com/question/317623081](https://www.zhihu.com/question/317623081)
  * 4.描述的是分布之间的距离而不是样本的距离。[https://blog.csdn.net/Mark_2018/article/details/105400648](https://blog.csdn.net/Mark_2018/article/details/105400648)

### （3）GAN的核心代码
```
    class Discriminator(keras.Model):

        def __init__(self):
            super(Discriminator, self).__init__()

            # [b, 64, 64, 3] => [b, 1]
            self.conv1 = layers.Conv2D(64, 5, 3, 'valid')

            self.conv2 = layers.Conv2D(128, 5, 3, 'valid')
            self.bn2 = layers.BatchNormalization()

            self.conv3 = layers.Conv2D(256, 5, 3, 'valid')
            self.bn3 = layers.BatchNormalization()

            # [b, h, w ,c] => [b, -1]
            self.flatten = layers.Flatten()
            self.fc = layers.Dense(1)


        def call(self, inputs, training=None):

            x = tf.nn.leaky_relu(self.conv1(inputs))
            x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
            x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))

            # [b, h, w, c] => [b, -1]
            x = self.flatten(x)
            # [b, -1] => [b, 1]
            logits = self.fc(x)

            return logits
            
    class Generator(keras.Model):

        def __init__(self):
            super(Generator, self).__init__()

            # z: [b, 100] => [b, 3*3*512] => [b, 3, 3, 512] => [b, 64, 64, 3]
            self.fc = layers.Dense(3*3*512)

            self.conv1 = layers.Conv2DTranspose(256, 3, 3, 'valid')
            self.bn1 = layers.BatchNormalization()

            self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
            self.bn2 = layers.BatchNormalization()

            self.conv3 = layers.Conv2DTranspose(3, 4, 3, 'valid')

        def call(self, inputs, training=None):
            # [z, 100] => [z, 3*3*512]
            x = self.fc(inputs)
            x = tf.reshape(x, [-1, 3, 3, 512])
            x = tf.nn.leaky_relu(x)

            #
            x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
            x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
            x = self.conv3(x)
            x = tf.tanh(x)

            return x

    def celoss_ones(logits):
        # [b, 1]
        # [b] = [1, 1, 1, 1,]
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,                #logits经sigmoid函数激活之后的交叉熵
                                      labels=tf.ones_like(logits))        #该操作返回一个具有和给定logits相同形状（shape）和相同数据类型（dtype），但是所有的元素都被设置为1的tensor

        return tf.reduce_mean(loss)
    
    
    def celoss_zeros(logits):
        # [b, 1]
        # [b] = [1, 1, 1, 1,]
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                      labels=tf.zeros_like(logits))      #该操作返回一个具有和给定logits相同形状（shape）和相同数据类型（dtype），但是所有的元素都被设置为0的tensor
        return tf.reduce_mean(loss)
    
    
    def g_loss_fn(generator, discriminator, batch_z, is_training):

        fake_image = generator(batch_z, is_training)
        d_fake_logits = discriminator(fake_image, is_training)
        loss = celoss_ones(d_fake_logits)

        return loss
    
    def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
        # 1. treat real image as real
        # 2. treat generated image as fake
        fake_image = generator(batch_z, is_training)
        d_fake_logits = discriminator(fake_image, is_training)
        d_real_logits = discriminator(batch_x, is_training)

        d_loss_real = celoss_ones(d_real_logits)
        d_loss_fake = celoss_zeros(d_fake_logits)

        loss = d_loss_fake + d_loss_real                    -----------------------------GAN loss

        return loss
```
### （4）WGAN的核心代码（对GAN）

```
    def gradient_penalty(discriminator, batch_x, fake_image):

        batchsz = batch_x.shape[0]

        # [b, h, w, c]
        t = tf.random.uniform([batchsz, 1, 1, 1])
        # [b, 1, 1, 1] => [b, h, w, c]
        t = tf.broadcast_to(t, batch_x.shape)

        interplate = t * batch_x + (1 - t) * fake_image                             #gp部分公式

        with tf.GradientTape() as tape:
            tape.watch([interplate])                                                #gp部分公式
            d_interplote_logits = discriminator(interplate)
        grads = tape.gradient(d_interplote_logits, interplate)

        # grads:[b, h, w, c] => [b, -1]
        grads = tf.reshape(grads, [grads.shape[0], -1])                             #gp部分公式
        gp = tf.norm(grads, axis=1) #[b]
        gp = tf.reduce_mean( (gp-1)**2 )

        return gp
    
    def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
        # 1. treat real image as real
        # 2. treat generated image as fake
        fake_image = generator(batch_z, is_training)
        d_fake_logits = discriminator(fake_image, is_training)
        d_real_logits = discriminator(batch_x, is_training)

        d_loss_real = celoss_ones(d_real_logits)
        d_loss_fake = celoss_zeros(d_fake_logits)
        gp = gradient_penalty(discriminator, batch_x, fake_image)                #wgan较gan的不同之处，gp

        loss = d_loss_fake + d_loss_real + 1. * gp              ---------------------------------WGAN loss

        return loss, gp

```
<br/>


## References:
* [深度学习与TensorFlow 2入门实战（完整版）](https://www.bilibili.com/video/BV1HV411q7xD?from=search&seid=14089320887830328110)---龙曲良
* [https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) ---[Joseph Rocca](https://medium.com/@joseph.rocca)
* [https://zhuanlan.zhihu.com/p/24767059](https://zhuanlan.zhihu.com/p/24767059)
* [https://github.com/hindupuravinash/the-gan-zoo](https://github.com/hindupuravinash/the-gan-zoo)
* [https://reiinakano.github.io/gan-playground/在线构建GAN](https://reiinakano.github.io/gan-playground/)
