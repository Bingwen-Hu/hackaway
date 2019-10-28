# 简读RelativisticGan

RelativisticGan 是2019年被 ICLR 录取的论文，一起来了解一下吧。


### 核心
Relativistic译为相对的。怎么个相对法呢？这要从标准Gan谈起。关于标准Gan，可以查看我[另一篇文章](./gan.md)。论文中，将标准Gan写成sGan（standard Gan)。sGan加入了本论中提到的特性，则变成了RsGan。在标准的Gan中，D是努力区分真品与赝品，使其对真品的判断始终为1，对赝品的判断始终为0。而RsGan的作者认为，由于D同时看到了真品与赝品，D只需要做到，能看出真品相对于赝品更真实即可。也就是说，D并不需要变成世界一流的鉴别师，它只要在真与假两种产品中，成功判断哪个是真的，哪个是假的即可。同样的，G也不需要做出特别仿真的产品，它只需要让D觉得假的相对于真的更真实即可。对G和D的要求同时下降了呀。

那么RGan系列模型表现如何？论文中提到，其训练更加稳定，大部分实验结果表示，其生成的图片更加真实。

### 一点代码
由于大部分代码与[这个](./gan.py)重复，这里只给出精要部分。
```py
# step 1: G generate fake data fx using noise z
z = torch.randn(batch_size, z_dim)
fx = G(z)
```
第一步，`G`使用原材料`z`制作假数据。

```py
# step 2: D judge on (fx, x), D should determine that x is more
# real than fx.
x = x.view(-1, x_dim)
real = D(x)
fake = D(fx)
D_loss = BCE_stable(real - fake, real_target)
D_optim.zero_grad()
D_loss.backward()
D_optim.step()
```
第二步，`real-fake`这部分表示真的比假的真了多少，这就是相对的真实。

```py
# step 3: G update, G should make fx more real than x
z = torch.randn(batch_size, z_dim)
fx = G(z)
fake = D(fx)
G_loss = BCE_stable(fake - real.detach(), real_target)
G_optim.zero_grad()
G_loss.backward()
G_optim.step()
```
第三步，`fake-real`这部分表示假的比真的真了多少。使用`detach`是为了将其从`D`的计算图中取出来。这是一个框架相关的细节，跟RGan无关。

### 这就是全部了！
RGan的思想也是非常直观，对公式推导感兴趣的可以去看原论文。