'''
单级：
    dwt():离散小波变换
    idwt():逆离散小波变换
'''
import pywt
x=[3,7,1,1,-2,5,4,6]#信号
#cA：相似系数，即低频成分 CD：细节系数，即高频成分
cA,cD=pywt.dwt(x,'haar')#用haar小波来进行离散小波变换

y=pywt.idwt(cA,cD,'haar')#逆离散小波变换,即重构回去
#print(cA)#[7.07106781 1.41421356 2.12132034 7.07106781]
#print(cD)#[-2.82842712  0.         -4.94974747 -1.41421356]
#print(y)#[ 3.  7.  1.  1. -2.  5.  4.  6.]

'''
多级：
    wavedec()
    waverec()
'''
#'wavelet'表示用小波变换，'sym'表示小波类型，level表示分解成几层
#coeffs=pywt.wavedec(x,'wavelet',mode='sym',level=n)#[cAn,cDn,cDn-1,...,cD2,cD1]=coeffs，从低到高
#y=pywt.waverec(coeffs,'wavelet',mode='sym')#重构
x=[3,7,1,1,-1,5,4,6,6,4,5,-2,1,1,7,3]#16
coeffs=pywt.wavedec(x,'db1',mode='periodic',level=2)
cA2,cD2,cD1=coeffs
y=pywt.waverec(coeffs,'db1',mode='periodic')
print("cA2=",cA2)#4，是cD1的一半
print("cD2=",cD2)#4，是cD1的一半
print("cD1=",cD1)#8，是x的一半
print("IDWT=",y)

'''
使用CWT进行1D信号去噪
'''
import numpy as np
import pywt
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt

x=pywt.data.ecg().astype(float)/256

sigma=0.05
x_noisy=x+sigma*np.random.randn(x.size)
#rescale_sigma表示是否对噪声的方差进行伸缩，denoise_wavelet默认可以处理一维和二维
x_denoize=denoise_wavelet(x_noisy,method='BayesShrink',mode='soft',wavelet_levels=3,wavelet='sym8',rescale_sigma='True')

plt.figure(figsize=(20,10),dpi=100)

plt.plot(x)
plt.show()

plt.plot(x_noisy)
plt.show()

plt.plot(x_denoize)
plt.show()

'''

'''
