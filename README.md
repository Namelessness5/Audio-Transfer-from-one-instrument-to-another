# Audio-Transfer-from-one-instrument-to-another
based on Pytorch, librosa and Pyworld.

# Ideas:
We figured out 5 methods:
1. Style GAN like the image
2. Aligned Neural Network  
3. non-negative matrix factorization
4. style-transfer like the image
5. tune recognition and replacement

Conclusion：
Method 2 works fine every time， but it's not satisfying. Method 5 is perfect at certain circumstance, where there is only one tune at the same time and it's easy to be recognition. The other methods are disappointing. 

# Method 1 (StarGAN-Audio):
  This is based on another repository in github (https://github.com/liusongxiang/StarGAN-Voice-Conversion).
  
  I tried the IRMAS dataset.
  
  My idea is to cut the audio to fit my network, but it will generate noise in the place I cut them.
  
  You can find my results in StarGAN-Audio/samples
  
  If you want to try on your own, run the preprocess.py first, and you can look up in the repository I refer to above.
  
# Method 2 (NN):
  The biggest difference between method 1 and 2 is not the network they use, it's the dataset. Method 2 requires the dataset is alligned (see NN/snG and NN/snP).
  
  If the dataset is aligned, we can change the input form (Assuming all the audios have been converted by stft, the shape should be like (n_fft, frame_num). It's 2-d, but if they are aligned, we can use the shape (n_fft, 1) as the input form), and the cut-noise problem won't bother anymore.
  
  So this idea can be stated like this: we have 1-d arrays with length of "n_fft" named xi, and we're trying to get a network F to satisfy the equation F(xi) = yi, where yi has the same shape of xi. It's easy to come out with matrix multiplication, which stands for linear layer. But I found it will yield strange sound. I thought it could be some tune which never appears in my dataset, so I used CNN instead (Translation Invariance, to achieve this, I also need to use cqt to replace the stft).
  
  To train on your own dataset, you still need to run preprocess.py first.
 
 # Method 3 (NMF):
 See NMF/audioTransform.py. Increase n_components, the converted audio gets more like the origin audio.
 
 # Method 4 (NMF/NeuralStyleTransfer.py)
 Based on another repository in github (https://github.com/alishdipani/Neural-Style-Transfer-Audio).
 
 # Method 5 (tune-r)
 If you want to run your own dataset, run split.py to split your audio into single tune. Then run the function genStyle in the convert.py, finally use the function convert.
 
 Notice: 
 
 1.The split.py doesn't work sometime, you can also use your own way to split or replace the function deviation in the split.py with the function in the convert.py.
 
 2. The function get_base_freq sometimes returns none because it doesn't know the base frequency, try some other method to identify the base frequency.
 
 # Last
 For more detail, read my code.
 
 Have fun and feel free to contact me at 2076690478@qq.com.
