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
* CelebA数据集生成结果（3个多小时，20epoch）
<img src="https://github.com/xiaoxiaokaiyan/New_Pytorch_WGAN_Celeba_Oxford102flowers_Anime/blob/main/result2_fake_images-norm-20.png" width = 50% height =50%  div align=center />

* Anime数据集生成结果（2个多小时，54epoch）
<img src="https://github.com/xiaoxiaokaiyan/New_Pytorch_WGAN_Celeba_Oxford102flowers_Anime/blob/main/result3_fake_images-norm-54.png" width = 50% height =50%  div align=center />

* Oxford_102_flowers数据集生成结果（4个多小时，694epoch）
<img src="https://github.com/xiaoxiaokaiyan/New_Pytorch_WGAN_Celeba_Oxford102flowers_Anime/blob/main/result1_fake_images-norm-694.png" width = 50% height =50% div align=center />
&nbsp;
<br/>


## Public Datasets:
* CelebFaces Attributes Dataset（CelebA）是一个香港中文大学的大型人脸属性数据集，拥有超过200K名人图像，每个图像都有40个属性注释。此数据集中的图像覆盖了大的姿势变化和背景杂乱。CelebA具有大量的多样性，大量的数量和丰富的注释，包括:10,177个身份，202,599个脸部图像，5个地标位置，每个图像40个二进制属性注释。该数据集可用作以下计算机视觉任务的训练和测试集：面部属性识别，面部检测和地标（或面部部分）定位。
  * dataset link:[http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* the Anime dataset should be prepared by yourself in ./data/faces/*.jpg,63565个彩色图片。
  * dataset link: [https://www.kaggle.com/splcher/animefacedataset](https://www.kaggle.com/splcher/animefacedataset)
* Oxford_102_flowers 是牛津大学在2009发布的图像数据集。包含102种英国常见花类，每个类别包含 40-258张图像。
<br/>

## Experience：
### （1）代码问题
```
      先运行data_processing.py，将文件夹下的图片变为统一像素，再通过wgan.py，通过dataset = datasets.ImageFolder('./', transform=trans)加载数据。
``` 
``` 
      dataset=torchvision.datasets.ImageFolder(
                       root, transform=None, --------------------------会加载root目录底下文件夹中的全部图片，且transform可自己定义
                       target_transform=None, 
                       loader=<function default_loader>, 
                       is_valid_file=None)
                       
      root：图片存储的根目录，即各类别文件夹所在目录的上一级目录。
      transform：对图片进行预处理的操作（函数），原始图片作为输入，返回一个转换后的图片。
      target_transform：对图片类别进行预处理的操作，输入为 target，输出对其的转换。如果不传该参数，即对 target 不做任何转换，返回的顺序索引 0,1, 2…
      loader：表示数据集加载方式，通常默认加载方式即可。
      is_valid_file：获取图像文件的路径并检查该文件是否为有效文件的函数(用于检查损坏文件)
          如：
                trans = transforms.Compose([
                                              transforms.Resize(64),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                          ])
                dataset = datasets.ImageFolder('./', transform=trans) 
```   

### （2）关于VAE和GAN的区别
  * 1.VAE和GAN都是目前来看效果比较好的生成模型，本质区别我觉得这是两种不同的角度，VAE希望通过一种显式(explicit)的方法找到一个概率密度，并通过最小化对数似函数的下限来得到最优解；
GAN则是对抗的方式来寻找一种平衡，不需要认为给定一个显式的概率密度函数。（李飞飞）
  * 2.简单来说，GAN和VAE都属于深度生成模型（deep generative models，DGM）而且属于implicit DGM。他们都能够从具有简单分布的随机噪声中生成具有复杂分布的数据（逼近真实数据分布），而两者的本质区别是从不同的视角来看待数据生成的过程，从而构建了不同的loss function作为衡量生成数据好坏的metric度量。
  * 3.要求得一个生成模型使其生成数据的分布 能够最小化与真实数据分布之间的某种分布差异度量，例如KL散度、JS散度、Wasserstein距离等。采用不同的差异度量会导出不同的loss function，比如KL散度会导出极大似然估计，JS散度会产生最原始GAN里的判别器，Wasserstein距离通过dual form会引入critic。而不同的深度生成模型，具体到GAN、VAE还是flow model，最本质的区别就是从不同的视角来看待数据生成的过程，从而采用不同的数据分布模型来表达。 [https://www.zhihu.com/question/317623081](https://www.zhihu.com/question/317623081)
  * 4.描述的是分布之间的距离而不是样本的距离。[https://blog.csdn.net/Mark_2018/article/details/105400648](https://blog.csdn.net/Mark_2018/article/details/105400648)


### （3）WGAN的核心代码

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
* [WGAN-GP训练流程](https://mathpretty.com/11133.html),[https://github.com/wmn7/ML_Practice/tree/master/2019_09_09](https://github.com/wmn7/ML_Practice/tree/master/2019_09_09)
* [深度学习与TensorFlow 2入门实战（完整版）](https://www.bilibili.com/video/BV1HV411q7xD?from=search&seid=14089320887830328110)---龙曲良
* [https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) ---[Joseph Rocca](https://medium.com/@joseph.rocca)
* [https://zhuanlan.zhihu.com/p/24767059](https://zhuanlan.zhihu.com/p/24767059)
* [https://github.com/hindupuravinash/the-gan-zoo](https://github.com/hindupuravinash/the-gan-zoo)
* [https://reiinakano.github.io/gan-playground/在线构建GAN](https://reiinakano.github.io/gan-playground/)
