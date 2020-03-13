import numpy as np
import matplotlib.pyplot as plt


BIT_NUM  =  1 * 10**4


def bit_to_QPSK(input_bit):
    bit_num = len(input_bit)
    input_sign = np.copy(input_bit)
    input_sign[input_sign == 0] = -1
    input_QPSK = (-input_sign[0:bit_num:2]) + 1j * (-input_sign[1:bit_num:2])
    input_QPSK = (1 / np.sqrt(2))*input_QPSK

    return input_QPSK


def QPSK_to_bit(input_QPSK):
    bit_num = len(input_QPSK)

    input_QPSK_copy = np.copy(input_QPSK)

    input_QPSK_copy = input_QPSK_copy*np.sqrt(2)

    even_bit = np.real(input_QPSK_copy)
    even_bit[even_bit > 0] = 0
    even_bit[even_bit < 0] = 1

    odd_bit = np.imag(input_QPSK_copy)
    odd_bit[odd_bit > 0] = 0
    odd_bit[odd_bit < 0] = 1

    output_bit = np.zeros(bit_num*2, dtype=int)
    output_bit[0:bit_num*2:2] = even_bit
    output_bit[1:bit_num*2:2] = odd_bit

    return output_bit

def channel_equalization(input_QPSK, h):
	QPSK = input_QPSK/h

	return QPSK


def bit_to_16QAM(input_bit):
    bit_num = len(input_bit)
    sym_num = bit_num / 4

    input_sign = np.ones(bit_num)
    input_sign[input_bit == 0] = -1

    temp = 2 * input_sign[0:bit_num:2] + input_sign[1:bit_num:2]

    temp_copy = temp.copy()
    temp[temp_copy == 3] = 1
    temp[temp_copy == 1] = 3

    TX_16QAM = (temp[0:int(sym_num * 2):2] - 1j * temp[1:int(sym_num * 2):2]) * 1 / np.sqrt(10)

    return TX_16QAM

def QAM16_to_bit(input_QAM16):
    sym_num = len(input_QAM16)
    bit_num = sym_num * 4

    input_QAM16 = input_QAM16 * np.sqrt(10)

    input_QAM16_real = np.real(input_QAM16) 
    input_QAM16_imag = -np.imag(input_QAM16)

    real_copy = input_QAM16_real.copy() 
    imag_copy = input_QAM16_imag.copy()

    input_QAM16_real[real_copy > 2] = 3
    input_QAM16_real[2 > real_copy] = 1
    input_QAM16_real[0 > real_copy] = -1
    input_QAM16_real[-2 > real_copy] = -3

    input_QAM16_imag[imag_copy > 2] = 3
    input_QAM16_imag[2 > imag_copy] = 1
    input_QAM16_imag[0 > imag_copy] = -1
    input_QAM16_imag[-2 > imag_copy] = -3

    temp = np.zeros(sym_num * 2, dtype=int) 
    temp[0:sym_num * 2:2] = input_QAM16_real 
    temp[1:sym_num * 2:2] = input_QAM16_imag

    temp2 = np.zeros(sym_num * 2, dtype=int) + 1j * np.zeros(sym_num * 2, dtype=int)
    temp2[temp == 3] = 1 - 0j
    temp2[temp == 1] = 1 + 1j
    temp2[temp == -1] = - 0 + 1j
    temp2[temp == -3] = - 0 - 0j

    input_bit = np.zeros(bit_num,dtype=int)
    input_bit[0:bit_num:2] = np.real(temp2)
    input_bit[1:bit_num:2] = np.imag(temp2)

    return input_bit




def wireless_gain(input_QPSK):
	bit_num = len(input_QPSK)
	h_i = np.random.normal(0, 1, int(bit_num))
	h_q = np.random.normal(0, 1, int(bit_num))
	h = (1 / np.sqrt(2))*(h_i + 1j*h_q)
	input_QPSK = h*input_QPSK

	return input_QPSK, h


def AWGN(input_QPSK, sigma2):
	bit_num = len(input_QPSK)
	n_i = np.random.normal(0, sigma2, int(bit_num))
	n_q = np.random.normal(0, sigma2, int(bit_num))
	n = (1 / np.sqrt(2))*(n_i + 1j*n_q)
	input_QPSK = input_QPSK + n

	return input_QPSK


def scatter_plot_sent_received(sent, received, limit):
	plt.scatter(received.real, received.imag, label="received", color="green", s=5)
	plt.scatter(sent.real, sent.imag, label="sent", color="red", s=50)
	plt.xlim(-limit, limit)
	plt.ylim(-limit, limit)
	plt.axhline(0, color='black')
	plt.axvline(0, color='black')
	plt.xlabel('real axis')
	plt.ylabel('imaginary axis')
	plt.title('complex numbers')
	plt.legend()
	plt.show()


def simulate(SNR):

	input_bit = np.random.randint(0, 2, BIT_NUM)

	input_QPSK = bit_to_QPSK(input_bit)
	output_QPSK, h = wireless_gain(input_QPSK)
	sigma2 = 1/SNR
	noise_QPSK = AWGN(output_QPSK, sigma2)
	noise_QPSK_noGain = channel_equalization(noise_QPSK, h)
	output_bit = QPSK_to_bit(noise_QPSK_noGain)

	if(SNR == 10):
		scatter_plot_sent_received(input_QPSK, noise_QPSK_noGain, 1.25)	
	elif(SNR == 1):
		scatter_plot_sent_received(input_QPSK, noise_QPSK_noGain, 2)
	elif(SNR == 0.1):
		scatter_plot_sent_received(input_QPSK, noise_QPSK_noGain, 3)


def simulate_without_plot(SNR):

	input_bit = np.random.randint(0, 2, BIT_NUM)

	input_QPSK = bit_to_QPSK(input_bit)
	output_QPSK, h = wireless_gain(input_QPSK)
	sigma2 = 1/SNR
	noise_QPSK = AWGN(output_QPSK, sigma2)
	noise_QPSK_noGain = channel_equalization(noise_QPSK, h)
	output_bit = QPSK_to_bit(noise_QPSK_noGain)

	error = ((sum(i != j for i, j in zip(input_bit, output_bit)))/BIT_NUM)*100

	return error

def error_base_SNR():
	error_list = []
	for i in np.arange(0.1, 10, 0.1):
		error_list.append(simulate_without_plot(i))
	plt.scatter(np.arange(0.1, 10, 0.1), error_list)
	plt.show()
	# print(sum(error_list))


def encode_hamming_4bit(input_bit):
	# the encoding matrix
	G = ['1101', '1011', '1000', '0111', '0100', '0010', '0001']
	p = ''.join(list(map(str, input_bit)))
	x = ''.join([str(bin(int(i, 2) & int(p, 2)).count('1') % 2) for i in G])
	return list(map(int, x))

def encode_hamming(input_bit):
	s2 = np.copy(input_bit)
	code = []
	for i in range(0, len(input_bit), 4):
		s = s2[:4]
		code.extend(encode_hamming_4bit(s))
		s2 = s2[4:]
	return code

def decode_hamming_7bit(input_bit):
	# the parity-check matrix
	H = ['1010101', '0110011', '0001111']
	Ht = ['100', '010', '110', '001', '101', '011', '111']
	# the decoding matrix
	R = ['0010000', '0000100', '0000010', '0000001']
	x = ''.join(list(map(str, input_bit)))
	z = ''.join([str(bin(int(j, 2) & int(x, 2)).count('1') % 2) for j in H])
	if int(z, 2) > 0:
	    e = int(Ht[int(z, 2) - 1], 2)
	else:
	    e = 0
	if e > 0:
	    x = list(x)
	    x[e - 1] = str(1 - int(x[e - 1]))
	    x = ''.join(x)
	p = ''.join([str(bin(int(k, 2) & int(x, 2)).count('1') % 2) for k in R])
	return list(map(int, p))

def decode_hamming(input_bit):
	s2 = np.copy(input_bit)
	code = []
	for i in range(0, len(input_bit), 7):
		s = s2[:7]
		code.extend(decode_hamming_7bit(s))
		s2 = s2[7:]
	return code


def simulate_hamming(SNR):

	input_bit = np.random.randint(0, 2, BIT_NUM)

	input_hamming = encode_hamming(input_bit)
	input_QPSK = bit_to_QPSK(input_hamming)
	output_QPSK, h = wireless_gain(input_QPSK)
	sigma2 = 1/SNR
	noise_QPSK = AWGN(output_QPSK, sigma2)
	noise_QPSK_noGain = channel_equalization(noise_QPSK, h)
	output_hamming = QPSK_to_bit(noise_QPSK_noGain)
	output_bit = decode_hamming(output_hamming)

	if(SNR == 10):
		scatter_plot_sent_received(input_QPSK, noise_QPSK_noGain, 1.25)	
	elif(SNR == 1):
		scatter_plot_sent_received(input_QPSK, noise_QPSK_noGain, 2)
	elif(SNR == 0.1):
		scatter_plot_sent_received(input_QPSK, noise_QPSK_noGain, 3)

def simulate_without_plot_hamming(SNR):

	input_bit = np.random.randint(0, 2, BIT_NUM)

	input_hamming = encode_hamming(input_bit)
	input_QPSK = bit_to_QPSK(input_hamming)
	output_QPSK, h = wireless_gain(input_QPSK)
	sigma2 = 1/SNR
	noise_QPSK = AWGN(output_QPSK, sigma2)
	noise_QPSK_noGain = channel_equalization(noise_QPSK, h)
	output_hamming = QPSK_to_bit(noise_QPSK_noGain)
	output_bit = decode_hamming(output_hamming)

	error = ((sum(i != j for i, j in zip(input_bit, output_bit)))/BIT_NUM)*100

	return error

def error_base_SNR_hamming():
	error_list = []
	for i in np.arange(0.1, 10, 0.1):
		error_list.append(simulate_without_plot_hamming(i))
	plt.scatter(np.arange(0.1, 10, 0.1), error_list)
	plt.show()
	# print(sum(error_list))


def simulate_16QAM(SNR):

	input_bit = np.random.randint(0, 2, BIT_NUM)

	input_QPSK = bit_to_16QAM(input_bit)
	output_QPSK, h = wireless_gain(input_QPSK)
	sigma2 = 1/SNR
	noise_QPSK = AWGN(output_QPSK, sigma2)
	noise_QPSK_noGain = channel_equalization(noise_QPSK, h)
	output_bit = QAM16_to_bit(noise_QPSK_noGain)

	if(SNR == 10):
		scatter_plot_sent_received(input_QPSK, noise_QPSK_noGain, 1.25)	
	elif(SNR == 1):
		scatter_plot_sent_received(input_QPSK, noise_QPSK_noGain, 2)
	elif(SNR == 0.1):
		scatter_plot_sent_received(input_QPSK, noise_QPSK_noGain, 3)


def simulate_without_plot_16QAM(SNR):

	input_bit = np.random.randint(0, 2, BIT_NUM)

	input_QPSK = bit_to_16QAM(input_bit)
	output_QPSK, h = wireless_gain(input_QPSK)
	sigma2 = 1/SNR
	noise_QPSK = AWGN(output_QPSK, sigma2)
	noise_QPSK_noGain = channel_equalization(noise_QPSK, h)
	output_bit = QAM16_to_bit(noise_QPSK_noGain)

	error = ((sum(i != j for i, j in zip(input_bit, output_bit)))/BIT_NUM)*100

	return error

def error_base_SNR_16QAM():
	error_list = []
	for i in np.arange(0.1, 10, 0.1):
		error_list.append(simulate_without_plot_16QAM(i))
	plt.scatter(np.arange(0.1, 10, 0.1), error_list)
	plt.show()
	# print(sum(error_list))



def simulate_normal():
	simulate(10)
	simulate(1)
	simulate(0.1)

	error_base_SNR()

def simulate_hamming_plots():
	simulate_hamming(10)
	simulate_hamming(1)
	simulate_hamming(0.1)

	error_base_SNR_hamming()

def simulate_16QAM_plots():
	simulate_16QAM(10)
	simulate_16QAM(1)
	simulate_16QAM(0.1)

	error_base_SNR_16QAM()



simulate_normal()
simulate_hamming_plots()
simulate_16QAM_plots()



