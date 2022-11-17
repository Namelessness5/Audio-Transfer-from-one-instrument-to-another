from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import copy
import librosa
import soundfile as sf
from tqdm import trange


class CNNModel(nn.Module):
		def __init__(self, in_channel):
			super(CNNModel, self).__init__()
			self.cnn1 = nn.Conv1d(in_channels=in_channel, out_channels=4096, kernel_size=3, stride=1, padding=1)
			self.cnn2 = nn.Conv1d(in_channels=4096, out_channels=4096, kernel_size=3, stride=1, padding=1)
			self.cnn = [self.cnn1, self.cnn2]
			#self.nl1 = nn.ReLU()
			#self.pool1 = nn.AvgPool1d(kernel_size=5)
			#self.fc1 = nn.Linear(4096*2500,2**5)
			#self.nl3 = nn.ReLU()
			#self.fc2 = nn.Linear(2**10,2**5)
		
		def forward(self, x):
			out = self.cnn1(x)
			out = self.cnn2(out)
			#out = self.nl1(out)
			#out = self.pool1(out)
			out = out.view(out.size(0),-1)
			#out = self.fc1(out)
			#out = self.nl3(out)
			#out = self.fc2(out)
			return out


class GramMatrix(nn.Module):

	def forward(self, input):
		a, b, c = input.size()  # a=batch size(=1)
		# b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
		features = input.view(a * b, c)  # resise F_XL into \hat F_XL
		G = torch.mm(features, features.t())  # compute the gram product
		# we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
		return G.div(a * b * c)


class StyleLoss(nn.Module):

	def __init__(self, target, weight):
		super(StyleLoss, self).__init__()
		self.target = target.detach() * weight
		self.weight = weight
		self.gram = GramMatrix()
		self.criterion = nn.MSELoss()

	def forward(self, input):
		self.output = input.clone()
		self.G = self.gram(input)
		self.G.mul_(self.weight)
		self.loss = self.criterion(self.G, self.target)
		return self.output

	def backward(self,retain_graph=True):
		self.loss.backward(retain_graph=retain_graph)
		return self.loss


def read_audio_spectum(filename):
	x, fs = librosa.load(filename)
	return x, fs

def get_style_model_and_losses(cnn, style_float, style_weight, style_layers):  # STYLE WEIGHT

	cnn = copy.deepcopy(cnn)
	style_losses = []
	model = nn.Sequential()  # the new Sequential module network
	gram = GramMatrix()  # we need a gram module in order to compute style targets
	if torch.cuda.is_available():
		model = model.cuda()
		gram = gram.cuda()

	for i in range(len(style_layers)):
		name = style_layers[i]
		model.add_module(name, cnn.cnn[i])
		target_feature = model(style_float).clone()
		target_feature_gram = gram(target_feature)
		style_loss = StyleLoss(target_feature_gram, style_weight)
		model.add_module("style_loss_{}".format(i), style_loss)
		style_losses.append(style_loss)
		# name = 'pool_1'
	# model.add_module(name, cnn.pool1)

		'''name = 'fc_1'
		model.add_module(name, cnn.fc1)

		name = 'nl_9'
		model.add_module(name, cnn.nl9)

		name = 'fc_2'
		model.add_module(name, cnn.fc2)'''
	return model, style_losses


def get_input_param_optimizer(input_float):
	input_param = nn.Parameter(input_float.data)
	#optimizer = optim.Adagrad([input_param], lr=learning_rate_initial, lr_decay=0.0001,weight_decay=0)
	optimizer = optim.Adam([input_param], lr=learning_rate_initial, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
	return input_param, optimizer


def run_style_transfer(cnn, style_float, input_float, num_steps, style_weight, style_layers): #STYLE WEIGHT, NUM_STEPS
	print('Building the style transfer model..')
	model, style_losses = get_style_model_and_losses(cnn, style_float, style_weight, style_layers)
	input_param, optimizer = get_input_param_optimizer(input_float)
	print('Optimizing..')
	run = [0]
	for i in range(num_steps):
	# while run[0] <= num_steps:
		def closure():
           	# correct the values of updated input image
			input_param.data.clamp_(0, 1)

			optimizer.zero_grad()
			model(input_param)
			style_score = 0

			for sl in style_losses:
					#print('sl is ',sl,' style loss is ',style_score)
				style_score += sl.backward()

			if (i + 1) % 100 == 0:
				print("run {}:".format(i))
				print('Style Loss : {:8f}'.format(style_score)) #CHANGE 4->8
				print()
				return style_score

		optimizer.step(closure)
	input_param.data.clamp_(0, 1)
	return input_param.data


learning_rate_initial = 0.03
num_steps = 600
style_weight = 2000
N_FFT = 2048
style_layers_default = ["conv_1"]
Method = "cqt"
# Method = "stft"


if __name__ == '__main__':
	#print('Enter the names of SCRIPT, Content audio, Style audio')
	content_audio_name = "audios/for~2.wav"
	style_audio_name = "audios/guitar.wav"

	style_audio, style_sr = read_audio_spectum(style_audio_name)
	content_audio, content_sr = read_audio_spectum(content_audio_name)
	length = np.max((len(style_audio), len(content_audio)))
	style_audio = np.pad(style_audio, (0, length-len(style_audio)), "constant")
	content_audio = np.pad(content_audio, (0, length-len(content_audio)), "constant")

	if Method == "stft":
		S = librosa.stft(style_audio, N_FFT)
		style_audio = np.log1p(np.abs(S))

		S = librosa.stft(content_audio, N_FFT)
		content_p = np.angle(S)
		content_audio = np.log1p(np.abs(S))

		num_samples = style_audio.shape[1]
		style_audio = style_audio.reshape([1, N_FFT // 2 + 1, num_samples])
		content_audio = content_audio.reshape([1, N_FFT // 2 + 1, num_samples])

	elif Method == "cqt":
		S = librosa.cqt(style_audio, hop_length=64)
		style_audio = np.abs(S)

		S = librosa.cqt(content_audio, hop_length=64)
		content_p = np.angle(S)
		content_audio = np.abs(S)

		num_samples = style_audio.shape[1]
		d2 = style_audio.shape[0]
		style_audio = style_audio.reshape([1, d2, num_samples])
		content_audio = content_audio.reshape([1, d2, num_samples])


	if(content_sr == style_sr):
		print('Sampling Rates are same')
	else:
		print('Sampling rates are not same')
		exit()

	if torch.cuda.is_available():
		style_float = Variable((torch.from_numpy(style_audio)).cuda())
		content_float = Variable((torch.from_numpy(content_audio)).cuda())	
	else:
		style_float = Variable(torch.from_numpy(style_audio))
		content_float = Variable(torch.from_numpy(content_audio))
	#style_float = style_float.unsqueeze(0)
	
	#style_float = style_float.view([1025,1,2500])
	
	'''
	print(style_float.size())
	exit()
	'''
	#style_float = style_float.unsqueeze(0)
	#content_float = content_float.unsqueeze(0)
	#content_float = content_float.reshape(1025,1,2500)
	
	#content_float = content_float.unsqueeze(0)
	#content_float = content_float.squeeze(0)

	if Method == "stft":
		cnn = CNNModel(N_FFT//2 + 1)
	elif Method == "cqt":
		cnn = CNNModel(84)

	if torch.cuda.is_available():
		cnn = cnn.cuda()

	input_float = content_float.clone()
	#input_float = Variable(torch.randn(content_float.size())).type(torch.FloatTensor)
		
	output = run_style_transfer(cnn, style_float, input_float, num_steps, style_weight, style_layers_default)
	if torch.cuda.is_available():
		output = output.cuda()

	#output = output.squeeze(0)
	output = output.squeeze(0)
	output = output.numpy()
	#print(output.shape)
	#output = output.resize([1025,2500])

	a = output

	# This code is supposed to do phase reconstruction
	if Method == "stft":
		p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
		for i in range(500):
			S = a * np.exp(1j*p)
			x = librosa.istft(S)
			p = np.angle(librosa.stft(x, N_FFT))

	elif Method == "cqt":
		S = a * np.exp(1j*content_p)
		x = librosa.icqt(S, hop_length=64)

	OUTPUT_FILENAME = "results/" + Method + str(num_steps)+'_c'+content_audio_name+'_rephase_'+style_audio_name+'_sw'+str(style_weight)+'.wav'
	sf.write(OUTPUT_FILENAME, x, content_sr, 'PCM_16')

	print('DONE...')
