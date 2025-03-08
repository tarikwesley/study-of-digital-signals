import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 1. PARAMETRIZAÇÕES
# ===========================
NUM_SYMBOLS = 5000           # Número de símbolos
NUM_CARRIERS = 64            # Subportadoras para OFDM
SNR_LEVELS = np.arange(0, 30, 2)  # Valores de SNR para análise
MODULATION_ORDERS = [2, 4, 8, 16] # Ordens de modulação PAM

# ===========================
# 2. GERAÇÃO DE DADOS
# ===========================
def generate_pam_data(order, num_symbols):
    data_bits = np.random.randint(0, order, size=num_symbols)
    return data_bits

def generate_ofdm_data(num_symbols, num_carriers):
    qam_bits = np.random.randint(0, 16, size=(num_symbols, num_carriers))
    return qam_bits

# ===========================
# 3. MODULAÇÃO
# ===========================
def modulate_pam(data_bits, order):
    levels = np.arange(-(order - 1), order, 2)  # Exemplo: [-3, -1, 1, 3] para 4-PAM
    symbols = levels[data_bits]
    return symbols, levels

def modulate_ofdm(qam_bits, num_carriers):
    qam_symbols = (2 * (qam_bits // 4) - 3) + 1j * (2 * (qam_bits % 4) - 3)
    ofdm_signal = np.fft.ifft(qam_symbols, axis=1).flatten()
    return qam_symbols, ofdm_signal

# ===========================
# 4. DEMODULAÇÃO
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
# 5. ADIÇÃO DE RUÍDO E CÁLCULO DE SNR
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
# 6. CÁLCULOS DE DESEMPENHO
# ===========================
def calculate_ber(original_bits, decoded_bits):
    errors = np.sum(original_bits != decoded_bits)
    return errors / len(original_bits)

# ===========================
# 7. SIMULAÇÕES
# ===========================
def simulate_pam(order, snr_levels, num_symbols):
    ber_per_snr = []
    
    for snr in snr_levels:
        # Geração e modulação
        data_bits = generate_pam_data(order, num_symbols)
        pam_symbols, pam_levels = modulate_pam(data_bits, order)
        
        # Adição de ruído
        noisy_signal = add_awgn_noise(pam_symbols, snr)
        actual_snr = calculate_snr(pam_symbols, noisy_signal)
        print(f"SNR desejado: {snr:.2f} dB, SNR real: {actual_snr:.2f} dB")
        
        # Demodulação e cálculo de BER
        decoded_bits = pam_demodulate(noisy_signal, pam_levels)
        ber = calculate_ber(data_bits, decoded_bits)
        ber_per_snr.append(ber)

    return ber_per_snr

def simulate_ofdm(snr_levels, num_symbols, num_carriers):
    ofdm_bers = []

    for snr in snr_levels:
        # Geração e modulação
        qam_bits = generate_ofdm_data(num_symbols, num_carriers)
        _, ofdm_signal = modulate_ofdm(qam_bits, num_carriers)

        # Adição de ruído
        noisy_signal = add_awgn_noise(ofdm_signal, snr)
        actual_snr = calculate_snr(ofdm_signal, noisy_signal)
        print(f"SNR desejado: {snr:.2f} dB, SNR real: {actual_snr:.2f} dB")

        # Demodulação e cálculo de BER
        decoded_bits = ofdm_demodulate(noisy_signal, num_symbols, num_carriers)
        ofdm_ber = calculate_ber(qam_bits.flatten(), decoded_bits)
        ofdm_bers.append(ofdm_ber)

    return ofdm_bers

# ===========================
# 8. VISUALIZAÇÕES
# ===========================
def plot_signal(signal, snr, ber):
    plt.figure(figsize=(10, 4))
    plt.plot(np.real(signal[:100]), label=f'OFDM com SNR {snr} dB\nBER: {ber:.2e}')
    plt.legend()
    plt.title(f'Sinal OFDM com SNR {snr} dB')
    plt.xlabel('Amostra')
    plt.ylabel('Amplitude')
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

# ===========================
# 9. EXECUÇÃO DAS SIMULAÇÕES
# ===========================
# Simulação PAM
ber_results_pam = {}
for order in MODULATION_ORDERS:
    print(f"Simulando para {order}-PAM")
    ber_results_pam[order] = simulate_pam(order, SNR_LEVELS, NUM_SYMBOLS)

# Simulação OFDM
print("Simulando OFDM")
ofdm_ber_results = simulate_ofdm(SNR_LEVELS, NUM_SYMBOLS, NUM_CARRIERS)

# Plotagem dos resultados
plot_ber(SNR_LEVELS, [ofdm_ber_results], labels=['OFDM'])
plot_ber(SNR_LEVELS, [ber_results_pam[order] for order in MODULATION_ORDERS], 
         labels=[f'{order}-PAM' for order in MODULATION_ORDERS])
