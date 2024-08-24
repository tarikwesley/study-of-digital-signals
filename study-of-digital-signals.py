import numpy as np
import matplotlib.pyplot as plt

# Parâmetros de simulação
num_symbols = 1000       # Número de símbolos
num_carriers = 64        # Número de subportadoras no OFDM
symbol_rate = 10e3       # Taxa de símbolos (Hz)
sampling_rate = 10 * symbol_rate  # Taxa de amostragem (Hz)
samples_per_symbol = int(sampling_rate // symbol_rate)  # Amostras por símbolo

# Geração de Símbolos Aleatórios para PAM:
pam_symbols = np.random.choice([-1, 1], size=num_symbols)

# Modulação PAM:
pam_signal = np.repeat(pam_symbols, samples_per_symbol)

# Geração de Símbolos Aleatórios para OFDM:
# Considerando modulação BPSK para OFDM (pode ser substituído por QAM)
ofdm_symbols = np.random.choice([-1, 1], size=(num_symbols, num_carriers))

# Modulação OFDM:
ofdm_signal = np.fft.ifft(ofdm_symbols, axis=1).flatten()

# Função para adicionar ruído AWGN:
def add_awgn_noise(signal, snr_dB):
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_dB / 10))
    noise = np.random.normal(scale=np.sqrt(noise_power/2), size=signal.shape) + 1j * np.random.normal(scale=np.sqrt(noise_power/2), size=signal.shape)
    noisy_signal = signal + noise
    return noisy_signal

# Funções para calcular BER e SNR:
def calculate_ber(original_bits, decoded_bits):
    errors = np.sum(original_bits != decoded_bits)
    ber = errors / len(original_bits)
    return ber

def calculate_snr(signal, noisy_signal):
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = np.mean(np.abs(noisy_signal - signal) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Definição de diferentes níveis de SNR:
snr_levels = [np.inf, 3, 5, 8, 10, 13, 15]
pam_bers = []
ofdm_bers = []

# Simulação e análise para diferentes níveis de SNR:
for snr in snr_levels:
    if np.isinf(snr):
        pam_noisy_signal = pam_signal.copy()
        ofdm_noisy_signal = ofdm_signal.copy()
        pam_ber = 0
        ofdm_ber = 0
    else:
        pam_noisy_signal = add_awgn_noise(pam_signal, snr)
        ofdm_noisy_signal = add_awgn_noise(ofdm_signal, snr)

        # Demodulação PAM
        pam_decoded = np.sign(np.real(pam_noisy_signal[::samples_per_symbol]))
        pam_decoded[pam_decoded == 0] = 1

        # Demodulação OFDM
        ofdm_noisy_symbols = np.fft.fft(ofdm_noisy_signal.reshape((num_symbols, num_carriers)), axis=1)
        ofdm_decoded = np.sign(ofdm_noisy_symbols.flatten().real)
        ofdm_decoded[ofdm_decoded == 0] = 1

        # Cálculo do BER
        pam_ber = calculate_ber(pam_symbols, pam_decoded)
        ofdm_ber = calculate_ber(ofdm_symbols.flatten(), ofdm_decoded)

    pam_bers.append(pam_ber)
    ofdm_bers.append(ofdm_ber)

    # Plotagem dos sinais ruidosos para PAM
    plt.figure(figsize=(10, 4))
    plt.plot(np.real(pam_noisy_signal[:100]), label=f'PAM com SNR {snr} dB\nBER: {pam_ber:.2e}')
    plt.legend()
    plt.title(f'Sinal PAM com SNR {snr} dB')
    plt.xlabel('Amostra')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plotagem dos sinais ruidosos para OFDM
    plt.figure(figsize=(10, 4))
    plt.plot(np.real(ofdm_noisy_signal[:100]), label=f'OFDM com SNR {snr} dB\nBER: {ofdm_ber:.2e}')
    plt.legend()
    plt.title(f'Sinal OFDM com SNR {snr} dB')
    plt.xlabel('Amostra')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.tight_layout()
    plt.show()

# Verificação de comprimento e plotagem final BER vs SNR:
if len(snr_levels) == len(pam_bers) == len(ofdm_bers):
    plt.figure(figsize=(10, 6))
    plt.plot(snr_levels, pam_bers, 'o-', label='PAM BER')
    plt.plot(snr_levels, ofdm_bers, 's-', label='OFDM BER')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('BER vs SNR')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
else:
    print("Erro: snr_levels e os valores de BER não têm o mesmo comprimento.")
