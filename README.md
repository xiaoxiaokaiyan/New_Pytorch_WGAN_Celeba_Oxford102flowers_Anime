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
```  
      出现：RuntimeError: invalid argument 0: Sizes of tensors must match except in dime
      这种错误有两种可能：
          1.你输入的图像数据的维度不完全是一样的，比如是训练的数据有100组，其中99组是256*256，但有一组是384*384，这样会导致Pytorch的检查程序报错。
          2.比较隐晦的batchsize的问题，Pytorch中检查你训练维度正确是按照每个batchsize的维度来检查的，比如你有1000组数据（假设每组数据为三通道256px*256px的图像），batchsize为4，那么每次训练             则提取(4,3,256,256)维度的张量来训练，刚好250个epoch解决(250*4=1000)。但是如果你有999组数据，你继续使用batchsize为4的话，这样999和4并不能整除，你在训练前249组时的张量维度都为               (4,3,256,256)但是最后一个批次的维度为(3,3,256,256)，Pytorch检查到(4,3,256,256) != (3,3,256,256)，维度不匹配，自然就会报错了，这可以称为一个小bug。
      解决办法：
          对于第一种：整理一下你的数据集保证每个图像的维度和通道数都一直即可。（本文的解决方法）
          对于第二种：挑选一个可以被数据集个数整除的batchsize或者直接把batchsize设置为1即可。

```  


### （2）关于VAE和GAN的区别
  * 1.VAE和GAN都是目前来看效果比较好的生成模型，本质区别我觉得这是两种不同的角度，VAE希望通过一种显式(explicit)的方法找到一个概率密度，并通过最小化对数似函数的下限来得到最优解；
GAN则是对抗的方式来寻找一种平衡，不需要认为给定一个显式的概率密度函数。（李飞飞）
  * 2.简单来说，GAN和VAE都属于深度生成模型（deep generative models，DGM）而且属于implicit DGM。他们都能够从具有简单分布的随机噪声中生成具有复杂分布的数据（逼近真实数据分布），而两者的本质区别是从不同的视角来看待数据生成的过程，从而构建了不同的loss function作为衡量生成数据好坏的metric度量。
  * 3.要求得一个生成模型使其生成数据的分布 能够最小化与真实数据分布之间的某种分布差异度量，例如KL散度、JS散度、Wasserstein距离等。采用不同的差异度量会导出不同的loss function，比如KL散度会导出极大似然估计，JS散度会产生最原始GAN里的判别器，Wasserstein距离通过dual form会引入critic。而不同的深度生成模型，具体到GAN、VAE还是flow model，最本质的区别就是从不同的视角来看待数据生成的过程，从而采用不同的数据分布模型来表达。 [https://www.zhihu.com/question/317623081](https://www.zhihu.com/question/317623081)
  * 4.描述的是分布之间的距离而不是样本的距离。[https://blog.csdn.net/Mark_2018/article/details/105400648](https://blog.csdn.net/Mark_2018/article/details/105400648)


### （3）WGAN的核心代码
  * 1.接着我们来定义网络, 我们首先定义分类器(discriminator), 这里我们是用来做动漫头像的分类.
```
   class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchN1 = nn.BatchNorm2d(64)
        self.LeakyReLU1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchN2 = nn.BatchNorm2d(64*2)
        self.LeakyReLU2 = nn.LeakyReLU(0.2, inplace=True)       

        self.conv3 = nn.Conv2d(in_channels=64*2, out_channels=64*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchN3 = nn.BatchNorm2d(64*4)
        self.LeakyReLU3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.Conv2d(in_channels=64*4, out_channels=64*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchN4 = nn.BatchNorm2d(64*8)
        self.LeakyReLU4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv5 = nn.Conv2d(in_channels=64*8, out_channels=1, kernel_size=4, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.LeakyReLU1(self.batchN1(self.conv1(x)))
        x = self.LeakyReLU2(self.batchN2(self.conv2(x)))
        x = self.LeakyReLU3(self.batchN3(self.conv3(x)))
        x = self.LeakyReLU4(self.batchN4(self.conv4(x)))
        x = self.conv5(x)
        return x
```

* 2.我们有的时候会测试一下我们的D是否是正确的, 于是我们可以从训练样本中抽取出一些来进行测试.
```
# 真实的图片
images = torch.stack(([dataset[i][0] for i in range(batch_size)]))
# 测试D是否与想象的是一样的
outputs = D(images)
```

* 3.接着我们定义生成器(generator), 生成器是输入随机数, 生成我们要模仿的动漫头像(Anime-Face).
```
  class Generator(nn.Module):
      def __init__(self):
          super(Generator, self).__init__()
          self.ConvT1 = nn.ConvTranspose2d(in_channels=100, out_channels=64*8, kernel_size=4, bias=False) # 这里的in_channels是和初始的随机数有关
          self.batchN1 = nn.BatchNorm2d(64*8)
          self.relu1 = nn.ReLU()

          self.ConvT2 = nn.ConvTranspose2d(in_channels=64*8, out_channels=64*4, kernel_size=4, stride=2, padding=1, bias=False) # 这里的in_channels是和初始的随机数有关
          self.batchN2 = nn.BatchNorm2d(64*4)
          self.relu2 = nn.ReLU()        

          self.ConvT3= nn.ConvTranspose2d(in_channels=64*4, out_channels=64*2, kernel_size=4, stride=2, padding=1, bias=False) # 这里的in_channels是和初始的随机数有关
          self.batchN3 = nn.BatchNorm2d(64*2)
          self.relu3 = nn.ReLU()

          self.ConvT4 = nn.ConvTranspose2d(in_channels=64*2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False) # 这里的in_channels是和初始的随机数有关
          self.batchN4 = nn.BatchNorm2d(64)
          self.relu4 = nn.ReLU()

          self.ConvT5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
          self.tanh = nn.Tanh() # 激活函数

      def forward(self, x):
          x = self.relu1(self.batchN1(self.ConvT1(x)))
          x = self.relu2(self.batchN2(self.ConvT2(x)))
          x = self.relu3(self.batchN3(self.ConvT3(x)))
          x = self.relu4(self.batchN4(self.ConvT4(x)))
          x = self.ConvT5(x)
          x = self.tanh(x)
          return x
          
```
* 4.同样的, 我们可以测试一下G是否是和我们想象中是一样进行工作的. 我们使用下面的方式进行测试.
```
  noise = Variable(torch.randn(batch_size, 100, 1, 1)).to(device) # 随机噪声，生成器输入
  # 测试G
  fake_images = G(noise)
```

* 5.加载数据集&定义辅助函数.
```
  trans = transforms.Compose([
          transforms.Resize(64),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])
  dataset = datasets.ImageFolder('./', transform=trans) # 数据路径
  dataloader = torch.utils.data.DataLoader(dataset,
                                          drop_last=True,
                                          batch_size=512, # 批量大小
                                          shuffle=False # 乱序  
                                          num_workers=2 # 多进程
                                          )
```

* 6.因为我们进行了归一化, 所以在图像最后进行保存的时候, 我们需要进行还原, 所以我们定义一个辅助函数来帮助进行还原.
```
  # 定义辅助函数
  def denorm(x):
      out = (x + 1) / 2
      return out.clamp(0, 1)
```

* 7.接着我们训练分类器(discriminator), 在训练WGAN-GP的discriminator的时候, 他是由三个部分的loss来组成的. 下面我们来每一步进行分解了进行查看.
* 7.1首先我们定义好要使用的real_label=1和fake_label=0, 和G需要使用的noise.
```
  batch_size = images.size(0)
  #images = images.reshape(batch_size, 3, 64, 64).to(device)
  mages = images.reshape(batch_size, 3, 64, 64).to(device)
  # 创造real label和fake label
  real_labels = torch.ones(batch_size, 1).to(device) # real的pic的label都是1
  fake_labels = torch.zeros(batch_size, 1).to(device) # fake的pic的label都是0
  noise = Variable(torch.randn(batch_size, 100, 1, 1)).to(device) # 随机噪声，生成器输入
```
  * 7.2接着我们计算loss的第一个组成部分(这里参考WGAN-GP的loss的计算公式).
```
  # 首先计算真实的图片的loss, d_loss_real
  outputs = D(images)
  d_loss_real = -torch.mean(outputs)
```
  * 7.3接着我们计算loss的第二个组成部分.
```
  # 接着计算假的图片的loss, d_loss_fake
  fake_images = G(noise)
  outputs = D(fake_images)
  d_loss_fake = torch.mean(outputs)
```
  * 7.4接着我们计算penalty region的loss, 也就是我们希望在penalty region中的梯度是越接近1越好,如上面图WGAN-Gradient-Penalty.
```
  # 接着计算penalty region 的loss, d_loss_penalty
  # 生成penalty region
  alpha = torch.rand((batch_size, 1, 1, 1)).to(device)
  x_hat = alpha * images.data + (1 - alpha) * fake_images.data
  x_hat.requires_grad = True
```
  * 7.5接着我们来计算他们的梯度, 我们希望梯度是越接近1越好.
```
  # 将中间的值进行分类
  pred_hat = D(x_hat)
  # 计算梯度
  gradient = torch.autograd.grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(device),
                     create_graph=False, retain_graph=False)
  # 这里的梯度计算完毕之后是在每一个像素点处都是有梯度的值的.计算出每一张图, 每一个像素点处的梯度
  gradient[0].shape
  """
  torch.Size([36, 3, 64, 64])
  """
```
  * 7.6接着我们计算L2范数.
```
  penalty_lambda = 10 # 梯度惩罚系数
  gradient_penalty = penalty_lambda * ((gradient[0].view(gradient[0].size()[0], -1).norm(p=2,dim=1)-1)**2).mean()
```
  * 7.7最后只需要把上面的三个部分相加, 进行反向传播来进行优化即可.
```
  # 三个loss相加, 反向传播进行优化
  d_loss = d_loss_real + d_loss_fake + gradient_penalty
  g_optimizer.zero_grad() # 两个优化器梯度都要清0
  d_optimizer.zero_grad()
  d_loss.backward()
  d_optimizer.step()
```
* 8.训练Generator
```
  normal_noise = Variable(torch.randn(batch_size, 100, 1, 1)).normal_(0, 1).to(device)
  fake_images = G(normal_noise) # 生成假的图片
  outputs = D(fake_images) # 放入辨别器
  g_loss = -torch.mean(outputs) # 希望生成器生成的图片判别器可以判别为真
  d_optimizer.zero_grad()
  g_optimizer.zero_grad()
  g_loss.backward()
  g_optimizer.step()
```
* 9.我们将上面的步骤重复N次, 反复训练D和G, 并将结果进行保存. 下面我们来看一下最后生成器生成的效果.首先我们导入已经训练好的模型.
```
  G = Generator().to(device) # 定义生成器
  # 读入生成器的模型
  G.load_state_dict(torch.load('./models/G.ckpt', map_location='cpu'))
  def show(img):
      """
      用来显示图片的
      """
      plt.figure(figsize=(24, 16))
      npimg = img.detach().numpy()
      plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
  # 使用生成器来进行生成
  test_noise = Variable(torch.FloatTensor(40, 100, 1, 1).normal_(0, 1)).to(device)
  fake_image = G(test_noise)
  show(make_grid(fake_image, nrow=8, padding=1, normalize=True, range=(-1, 1), scale_each=False, pad_value=0.5))
```
* 10.随机取出两个图片.
```
  test_noise = Variable(torch.FloatTensor(2, 100, 1, 1).normal_(0, 1)).to(device)
  fake_image = G(test_noise)
  show(make_grid(fake_image, nrow=2, padding=1, normalize=True, range=(-1, 1), scale_each=False, pad_value=0.5))
```
<br/>


## References:
* [WGAN-GP训练流程---对本代码的详细讲解](https://mathpretty.com/11133.html)
* [https://github.com/wmn7/ML_Practice/tree/master/2019_09_09](https://github.com/wmn7/ML_Practice/tree/master/2019_09_09)
* [RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0. Got 544 and 1935 in dimension 2 at ../aten/src/TH/generic/THTensor.cpp:711](https://www.cnblogs.com/zxj9487/p/11531888.html)
* [PyTorch修炼二、带你详细了解并使用Dataset以及DataLoader](https://zhuanlan.zhihu.com/p/128679151)

* [深度学习与TensorFlow 2入门实战（完整版）](https://www.bilibili.com/video/BV1HV411q7xD?from=search&seid=14089320887830328110)---龙曲良
* [https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) ---[Joseph Rocca](https://medium.com/@joseph.rocca)
* [https://zhuanlan.zhihu.com/p/24767059](https://zhuanlan.zhihu.com/p/24767059)
* [https://github.com/hindupuravinash/the-gan-zoo](https://github.com/hindupuravinash/the-gan-zoo)
* [https://reiinakano.github.io/gan-playground/在线构建GAN](https://reiinakano.github.io/gan-playground/)
