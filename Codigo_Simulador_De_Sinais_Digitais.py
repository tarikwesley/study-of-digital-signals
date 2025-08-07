import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 1. PARAMETRIZAÇÃO GERAL
# ===========================
NUM_SYMBOLS = 100000
NUM_CARRIERS = 64
SNR_LEVELS = np.arange(0, 30, 2)
MODULATION_ORDERS = [2, 4, 8, 16]

# ---
# 2. FUNÇÕES DE PROCESSAMENTO DE DADOS
# ---

# Geração de dados
def generate_pam_data(order, num_symbols):
    data_bits = np.random.randint(0, order, size=num_symbols)
    return data_bits

def generate_ofdm_data(num_symbols, num_carriers):
    qam_bits = np.random.randint(0, 16, size=(num_symbols, num_carriers))
    return qam_bits

# Modulação
def modulate_pam(data_bits, order):
    levels = np.arange(-(order - 1), order, 2)
    symbols = levels[data_bits]
    return symbols, levels

def modulate_ofdm(qam_bits, num_carriers):
    qam_symbols = (2 * (qam_bits // 4) - 3) + 1j * (2 * (qam_bits % 4) - 3)
    ofdm_signal = np.fft.ifft(qam_symbols, axis=1).flatten()
    return qam_symbols, ofdm_signal

# Demodulação
def pam_demodulate(received_signal, levels):
    decisions = np.argmin(np.abs(received_signal[:, None] - levels), axis=1)
    return decisions

def ofdm_demodulate(noisy_signal, num_symbols, num_carriers):
    noisy_symbols = np.fft.fft(noisy_signal.reshape((num_symbols, num_carriers)), axis=1)
    decoded_simulated = np.round((noisy_symbols.real + 3) / 2).astype(int).clip(0, 3)
    decoded_imag = np.round((noisy_symbols.imag + 3) / 2).astype(int).clip(0, 3)
    decoded_bits = (decoded_simulated * 4 + decoded_imag).flatten()
    return decoded_bits

# Ruído e SNR
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

# Desempenho
def calculate_ber(original_bits, decoded_bits):
    errors = np.sum(original_bits != decoded_bits)
    ber = errors / len(original_bits)
    return ber if ber > 0 else np.nan

# ---
# 3. FUNÇÕES DE SIMULAÇÃO
# ---

def simulate_pam(order, snr_levels, num_symbols):
    ber_per_snr = []
    snr_simulated_values = []

    for snr in snr_levels:
        data_bits = generate_pam_data(order, num_symbols)
        pam_symbols, pam_levels = modulate_pam(data_bits, order)
        noisy_signal = add_awgn_noise(pam_symbols, snr)
        actual_snr = calculate_snr(pam_symbols, noisy_signal)
        snr_simulated_values.append(actual_snr)
        decoded_bits = pam_demodulate(noisy_signal, pam_levels)
        ber = calculate_ber(data_bits, decoded_bits)
        ber_per_snr.append(ber)

    return ber_per_snr, snr_simulated_values

def simulate_ofdm(snr_levels, num_symbols, num_carriers):
    ofdm_bers = []
    snr_simulated_values = []

    for snr in snr_levels:
        qam_bits = generate_ofdm_data(num_symbols, num_carriers)
        _, ofdm_signal = modulate_ofdm(qam_bits, num_carriers)
        noisy_signal = add_awgn_noise(ofdm_signal, snr)
        actual_snr = calculate_snr(ofdm_signal, noisy_signal)
        snr_simulated_values.append(actual_snr)
        decoded_bits = ofdm_demodulate(noisy_signal, num_symbols, num_carriers)
        ofdm_ber = calculate_ber(qam_bits.flatten(), decoded_bits)
        ofdm_bers.append(ofdm_ber)

    return ofdm_bers, snr_simulated_values

# ---
# 4. FUNÇÕES DE VISUALIZAÇÃO
# ---

def plot_signal_evolution(original_signal, noisy_signal, demodulated_signal, titulo):
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(np.real(original_signal[:100]), label='Original', color='blue', linestyle='-')
    plt.plot(np.real(noisy_signal[:100]), label='Com Ruído', color='red', alpha=0.7, linestyle='--')
    plt.title(f'{titulo} - Sinal Original vs Com Ruído')
    plt.xlabel('Amostras')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(np.real(demodulated_signal[:100]), label='Demodulado', color='green', linestyle='-')
    plt.title(f'{titulo} - Sinal Demodulado')
    plt.xlabel('Amostras')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_snr_comparison(snr_levels, snr_simulated_values, title):
    plt.figure(figsize=(10, 6))
    plt.plot(snr_levels, snr_levels, 'r--', label='SNR Desejado')
    plt.plot(snr_levels, snr_simulated_values, 'bo-', label='SNR Simulado')
    plt.xlabel('SNR Desejado (dB)')
    plt.ylabel('SNR (dB)')
    plt.title(f'{title} - SNR Desejado vs Simulado')
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

# ---
# 5. EXECUÇÃO PRINCIPAL
# ---

def main():
    # Execução das simulações
    ber_results_pam = {}
    snr_simulated_pam = {}
    for order in MODULATION_ORDERS:
        ber_results_pam[order], snr_simulated_pam[order] = simulate_pam(order, SNR_LEVELS, NUM_SYMBOLS)
    ofdm_ber_results, ofdm_snr_simulated = simulate_ofdm(SNR_LEVELS, NUM_SYMBOLS, NUM_CARRIERS)

    # Visualização de exemplo de sinal (SNR fixo)
    snr_example = 10
    data_bits_pam = generate_pam_data(4, NUM_SYMBOLS)
    pam_symbols, pam_levels = modulate_pam(data_bits_pam, 4)
    noisy_signal_pam = add_awgn_noise(pam_symbols, snr_example)
    decoded_bits_pam = pam_demodulate(noisy_signal_pam, pam_levels)
    plot_signal_evolution(pam_symbols, noisy_signal_pam, decoded_bits_pam, f'{4}-PAM')

    qam_bits_ofdm = generate_ofdm_data(NUM_SYMBOLS, NUM_CARRIERS)
    _, ofdm_signal = modulate_ofdm(qam_bits_ofdm, NUM_CARRIERS)
    noisy_signal_ofdm = add_awgn_noise(ofdm_signal, snr_example)
    decoded_bits_ofdm = ofdm_demodulate(noisy_signal_ofdm, NUM_SYMBOLS, NUM_CARRIERS)
    plot_signal_evolution(ofdm_signal, noisy_signal_ofdm, decoded_bits_ofdm, 'OFDM')

    # Plotagem dos resultados de SNR
    for order in MODULATION_ORDERS:
        plot_snr_comparison(SNR_LEVELS, snr_simulated_pam[order], title=f'{order}-PAM')
    plot_snr_comparison(SNR_LEVELS, ofdm_snr_simulated, title='OFDM')

    # Plotagem dos resultados de BER
    plot_ber(SNR_LEVELS, [ofdm_ber_results], labels=['OFDM'])
    plot_ber(SNR_LEVELS, [ber_results_pam[order] for order in MODULATION_ORDERS],
             labels=[f'{order}-PAM' for order in MODULATION_ORDERS])

if __name__ == '__main__':
    main()