import numpy as np
import matplotlib.pyplot as plt

# ===========================
# PARAMETRIZAÇÕES
# ===========================
NUM_SYMBOLS = 100000           # Número de símbolos
NUM_CARRIERS = 64            # Subportadoras para OFDM
SNR_LEVELS = np.arange(0, 30, 2)  # Valores de SNR para análise
MODULATION_ORDERS = [2, 4, 8, 16] # Ordens de modulação PAM

# ===========================
# GERAÇÃO DE DADOS
# ===========================
def generate_pam_data(order, num_symbols):
    data_bits = np.random.randint(0, order, size=num_symbols)
    return data_bits

def generate_ofdm_data(num_symbols, num_carriers):
    qam_bits = np.random.randint(0, 16, size=(num_symbols, num_carriers))
    return qam_bits

# ===========================
# MODULAÇÃO
# ===========================
def modulate_pam(data_bits, order):
    levels = np.arange(-(order - 1), order, 2)
    symbols = levels[data_bits]
    return symbols, levels

def modulate_ofdm(qam_bits, num_carriers):
    qam_symbols = (2 * (qam_bits // 4) - 3) + 1j * (2 * (qam_bits % 4) - 3)
    ofdm_signal = np.fft.ifft(qam_symbols, axis=1).flatten()
    return qam_symbols, ofdm_signal

# ===========================
# DEMODULAÇÃO
# ===========================
def pam_demodulate(received_signal, levels):
    decisions = np.argmin(np.abs(received_signal[:, None] - levels), axis=1)
    return decisions

def ofdm_demodulate(noisy_signal, num_symbols, num_carriers):
    noisy_symbols = np.fft.fft(noisy_signal.reshape((num_symbols, num_carriers)), axis=1)
    decoded_real = np.round((noisy_symbols.real + 3) / 2).astype(int).clip(0, 3)
    decoded_imag = np.round((noisy_symbols.imag + 3) / 2).astype(int).clip(0, 3)
    decoded_bits = (decoded_real * 4 + decoded_imag).flatten()
    return decoded_bits

# ===========================
# ADIÇÃO DE RUÍDO E CÁLCULO DE SNR
# ===========================
def add_awgn_noise(signal, snr_dB):
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_dB / 10))
    noise = np.random.normal(scale=np.sqrt(noise_power), size=signal.shape)
    return signal + noise

def calculate_snr(signal, noisy_signal):
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = np.mean(np.abs(noisy_signal - signal) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# ===========================
# CÁLCULOS DE DESEMPENHO
# ===========================
def calculate_ber(original_bits, decoded_bits):
    errors = np.sum(original_bits != decoded_bits)
    ber = errors / len(original_bits)
    return ber

# ===========================
# SIMULAÇÕES
# ===========================
def simulate_pam(order, snr_levels, num_symbols):
    ber_per_snr = []
    snr_real_values = []

    for snr in snr_levels:
        # Geração e modulação
        data_bits = generate_pam_data(order, num_symbols)
        pam_symbols, pam_levels = modulate_pam(data_bits, order)

        # Adição de ruído
        noisy_signal = add_awgn_noise(pam_symbols, snr)
        actual_snr = calculate_snr(pam_symbols, noisy_signal)
        snr_real_values.append(actual_snr)
        
        # Demodulação e cálculo de BER
        decoded_bits = pam_demodulate(noisy_signal, pam_levels)
        ber = calculate_ber(data_bits, decoded_bits)
        ber_per_snr.append(ber)

    return ber_per_snr, snr_real_values

def simulate_ofdm(snr_levels, num_symbols, num_carriers):
    ofdm_bers = []
    snr_real_values = []

    for snr in snr_levels:
        # Geração e modulação
        qam_bits = generate_ofdm_data(num_symbols, num_carriers)
        _, ofdm_signal = modulate_ofdm(qam_bits, num_carriers)

        # Adição de ruído
        noisy_signal = add_awgn_noise(ofdm_signal, snr)
        actual_snr = calculate_snr(ofdm_signal, noisy_signal)
        snr_real_values.append(actual_snr)

        # Demodulação e cálculo de BER
        decoded_bits = ofdm_demodulate(noisy_signal, num_symbols, num_carriers)
        ofdm_ber = calculate_ber(qam_bits.flatten(), decoded_bits)
        ofdm_bers.append(ofdm_ber)

    return ofdm_bers, snr_real_values

# ===========================
# EXECUÇÃO DAS SIMULAÇÕES
# ===========================
# Simulação PAM
ber_results_pam = {}
snr_real_pam = {}

for order in MODULATION_ORDERS:
    ber_results_pam[order], snr_real_pam[order] = simulate_pam(order, SNR_LEVELS, NUM_SYMBOLS)

# Simulação OFDM
ofdm_ber_results, ofdm_snr_real = simulate_ofdm(SNR_LEVELS, NUM_SYMBOLS, NUM_CARRIERS)

# ===========================
# VISUALIZAÇÕES
# ===========================
def plot_signal_evolution(original_signal, noisy_signal, modulated_signal, demodulated_signal, titulo):
    plt.figure(figsize=(14, 10))  # Aumentando o tamanho da figura

    # Sinal original
    plt.subplot(4, 1, 1)
    plt.plot(np.real(original_signal[:100]), label='Original')
    plt.title(f'{titulo} - Sinal Original')
    plt.legend()
    plt.grid()

    # Sinal com ruído
    plt.subplot(4, 1, 2)
    plt.plot(np.real(noisy_signal[:100]), label='Com Ruído')
    plt.title(f'{titulo} - Sinal com Ruído')
    plt.legend()
    plt.grid()

    # Sinal modulado
    plt.subplot(4, 1, 3)
    plt.plot(np.real(modulated_signal[:100]), label='Modulado')
    plt.title(f'{titulo} - Sinal Modulado')
    plt.legend()
    plt.grid()

    # Sinal demodulado
    plt.subplot(4, 1, 4)
    plt.plot(np.real(demodulated_signal[:100]), label='Demodulado')
    plt.title(f'{titulo} - Sinal Demodulado')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_snr_comparison(snr_levels, snr_real_values, title):
    plt.figure(figsize=(10, 6))
    plt.plot(snr_levels, snr_levels, 'r--', label='SNR Desejado')
    plt.plot(snr_levels, snr_real_values, 'bo-', label='SNR Real')
    plt.xlabel('SNR Desejado (dB)')
    plt.ylabel('SNR (dB)')
    plt.title(f'{title} - SNR Desejado vs Real')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_ber(snr_levels, ber_values, labels):
    plt.figure(figsize=(10, 6))
    for ber, label in zip(ber_values, labels):
        plt.plot(snr_levels, ber, 'o-', label=label)

    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.title('BER vs SNR')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


snr_example = 10  # SNR de exemplo para visualização

data_bits = generate_pam_data(4, NUM_SYMBOLS)
pam_symbols, pam_levels = modulate_pam(data_bits, 4)
noisy_signal = add_awgn_noise(pam_symbols, snr_example)  # Exemplo com SNR fixo para visualização
decoded_bits = pam_demodulate(noisy_signal, pam_levels)

# Chama a função de plot com sinais apropriados
plot_signal_evolution(pam_symbols, noisy_signal, pam_symbols, decoded_bits, f'{4}-PAM')

qam_bits = generate_ofdm_data(NUM_SYMBOLS, NUM_CARRIERS)
_, ofdm_signal = modulate_ofdm(qam_bits, NUM_CARRIERS)
noisy_signal = add_awgn_noise(ofdm_signal, snr_example)  # SNR fixo para visualização
decoded_bits = ofdm_demodulate(noisy_signal, NUM_SYMBOLS, NUM_CARRIERS)

plot_signal_evolution(ofdm_signal, noisy_signal, ofdm_signal, decoded_bits, 'OFDM')



# Plotagem dos resultados de SNR
for order in MODULATION_ORDERS:
    plot_snr_comparison(SNR_LEVELS, snr_real_pam[order], title=f'{order}-PAM')
plot_snr_comparison(SNR_LEVELS, ofdm_snr_real, title='OFDM')

# Plotagem dos resultados de BER
plot_ber(SNR_LEVELS, [ofdm_ber_results], labels=['OFDM'])
plot_ber(SNR_LEVELS, [ber_results_pam[order] for order in MODULATION_ORDERS], 
         labels=[f'{order}-PAM' for order in MODULATION_ORDERS])

# Plotagem combinada dos resultados de BER
ber_values = [ofdm_ber_results] + [ber_results_pam[order] for order in MODULATION_ORDERS]
labels = ['OFDM'] + [f'{order}-PAM' for order in MODULATION_ORDERS]

plot_ber(SNR_LEVELS, ber_values, labels=labels)