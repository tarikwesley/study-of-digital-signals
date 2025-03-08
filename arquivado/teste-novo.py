import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
num_bits = 10000  # Número de bits a serem transmitidos
M = 4  # Número de níveis de modulação (4-QAM)
SNR_dB_range = np.arange(0, 21, 2)  # Faixa de SNR em dB
N = 64  # Número de subportadoras OFDM
CP = 16  # Comprimento do prefixo cíclico

# Função para gerar símbolos QAM
def generate_qam_symbols(bits, M):
    bits_reshaped = bits.reshape(-1, int(np.log2(M)))
    symbols = np.zeros(len(bits_reshaped), dtype=complex)
    for i, bit_group in enumerate(bits_reshaped):
        if M == 4:  # 4-QAM (QPSK)
            symbols[i] = (2 * bit_group[0] - 1) + 1j * (2 * bit_group[1] - 1)
    return symbols / np.sqrt(2)  # Normaliza a energia

# Função para adicionar ruído ao sinal
def add_noise(signal, SNR_dB):
    SNR = 10 ** (SNR_dB / 10)
    noise_power = 1 / SNR
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

# Função para calcular BER
def calculate_ber(received_bits, original_bits):
    return np.sum(received_bits != original_bits) / len(original_bits)

# Simulação PAM
ber_pam = []
for SNR_dB in SNR_dB_range:
    bits = np.random.randint(0, 2, num_bits)
    pam_signal = 2 * bits - 1  # BPSK (1 bit por símbolo)
    noisy_signal = add_noise(pam_signal, SNR_dB)
    received_bits = np.round((noisy_signal + 1) / 2).astype(int)
    ber_pam.append(calculate_ber(received_bits, bits))

# Simulação OFDM
ber_ofdm = []
for SNR_dB in SNR_dB_range:
    bits = np.random.randint(0, 2, num_bits)
    
    # Modulação QAM
    qam_symbols = generate_qam_symbols(bits, M)
    
    # Divisão em blocos OFDM
    num_symbols = len(qam_symbols)
    num_ofdm_blocks = int(np.ceil(num_symbols / N))
    qam_symbols_padded = np.zeros(num_ofdm_blocks * N, dtype=complex)
    qam_symbols_padded[:num_symbols] = qam_symbols
    
    # IFFT para gerar o sinal OFDM no domínio do tempo
    ofdm_symbols = np.fft.ifft(qam_symbols_padded.reshape(-1, N))
    
    # Adicionar prefixo cíclico
    ofdm_symbols_cp = np.hstack([ofdm_symbols[:, -CP:], ofdm_symbols])
    
    # Transmissão (adicionar ruído)
    noisy_ofdm_symbols = add_noise(ofdm_symbols_cp.flatten(), SNR_dB)
    
    # Remover prefixo cíclico
    noisy_ofdm_symbols = noisy_ofdm_symbols.reshape(-1, N + CP)[:, CP:]
    
    # FFT para voltar ao domínio da frequência
    received_qam_symbols = np.fft.fft(noisy_ofdm_symbols).flatten()[:num_symbols]
    
    # Demodulação QAM
    received_bits = np.zeros_like(bits)
    for i, symbol in enumerate(received_qam_symbols):
        received_bits[2 * i] = int(np.real(symbol) > 0)
        received_bits[2 * i + 1] = int(np.imag(symbol) > 0)
    
    # Calcular BER
    ber_ofdm.append(calculate_ber(received_bits, bits))

# Plotando os resultados
plt.figure(figsize=(10, 6))
plt.semilogy(SNR_dB_range, ber_pam, 'bo-', label='PAM (BPSK)')
plt.semilogy(SNR_dB_range, ber_ofdm, 'ro-', label='OFDM (4-QAM)')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('Comparação de BER entre PAM e OFDM')
plt.grid(True)
plt.legend()
plt.show()