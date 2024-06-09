import numpy as np
import matplotlib.pyplot as plt

num_symbols = 1000       # Número de símbolos
num_carriers = 64        # Número de subportadoras no OFDM
symbol_rate = 10e3       # Taxa de símbolos em símbolos por segundo (Hz)
sampling_rate = 10 * symbol_rate  # Taxa de amostragem em amostras por segundo (Hz)

# Geração de Símbolos Aleatórios para PAM:
pam_symbols = np.random.randint(0, 2, size=num_symbols) * 2 - 1

# Modulação PAM:
pam_signal = np.repeat(pam_symbols, sampling_rate // symbol_rate)

# Geração de Símbolos Aleatórios para OFDM:
ofdm_symbols = np.random.randint(0, 2, size=(num_symbols, num_carriers)) * 2 - 1

# Modulação OFDM:
ofdm_signal = np.fft.ifft(ofdm_symbols, axis=1)
ofdm_signal = ofdm_signal.flatten()

# Adição de Ruído aos Sinais:
def add_awgn_noise(signal, snr_dB):
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_dB / 10))
    noise = np.random.normal(scale=np.sqrt(noise_power), size=len(signal))
    noisy_signal = signal + noise
    return noisy_signal

# Função que calcula BER e função que calcula SNR
def calculate_ber(original_bits, noisy_bits):
    errors = np.sum(original_bits != noisy_bits)
    ber = errors / len(original_bits)
    return ber

def calculate_snr(signal, noisy_signal):
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = np.mean(np.abs(noisy_signal - signal) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Definição de Diferentes Níveis de SNR:
snr_levels = [np.inf, 5, 8, 10, 15]
pam_bers = []
ofdm_bers = []

# Plotagem dos Sinais para Diferentes Níveis de SNR. 
plt.figure(figsize=(10, 14))
for i, snr in enumerate(snr_levels):
    pam_noisy_signal = add_awgn_noise(pam_signal, snr)
    ofdm_noisy_signal = add_awgn_noise(ofdm_signal, snr)

    # Decodifica sinais ruidosos
    pam_decoded = np.sign(pam_noisy_signal[::int(sampling_rate // symbol_rate)])
    pam_decoded[pam_decoded == 0] = 1

    ofdm_noisy_symbols = np.fft.fft(ofdm_noisy_signal.reshape((num_symbols, num_carriers)), axis=1)
    ofdm_decoded = np.sign(ofdm_noisy_symbols.flatten())
    ofdm_decoded[ofdm_decoded == 0] = 1

    # Calcula BER
    pam_ber = calculate_ber(pam_symbols, pam_decoded)
    ofdm_ber = calculate_ber(ofdm_symbols.flatten(), ofdm_decoded)

    pam_bers.append(pam_ber)
    ofdm_bers.append(ofdm_ber)

    # Calcula SNR real
    pam_snr = calculate_snr(pam_signal, pam_noisy_signal)
    ofdm_snr = calculate_snr(ofdm_signal, ofdm_noisy_signal)

    # Plota sinais ruidosos
    plt.subplot(len(snr_levels) + 1, 2, 2*i+1)
    plt.plot(pam_noisy_signal[:100], label=f'PAM com SNR {snr} dB\nBER: {pam_ber:.2e}')
    plt.legend()

    plt.subplot(len(snr_levels) + 1, 2, 2*i+2)
    plt.plot(ofdm_noisy_signal[:100], label=f'OFDM com SNR {snr} dB\nBER: {ofdm_ber:.2e}')
    plt.legend()

# Plota BER vs SNR
plt.subplot(len(snr_levels) + 1, 1, len(snr_levels) + 1)
plt.plot(snr_levels, pam_bers, 'o-', label='PAM BER')
plt.plot(snr_levels, ofdm_bers, 's-', label='OFDM BER')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER vs SNR')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()