import numpy as np
import matplotlib.pyplot as plt

def generate_pam_symbols(M, num_symbols):
    """Gera símbolos PAM normalizados com média 0 e energia unitária."""
    symbols = np.arange(M)
    symbols = 2 * symbols - (M - 1)  # Centraliza em torno de 0
    symbols = symbols / np.sqrt(np.mean(symbols**2))  # Normaliza energia para 1
    return np.random.choice(symbols, num_symbols)

def add_awgn_noise(signal, snr_db):
    """Adiciona ruído AWGN ao sinal dado um SNR em dB."""
    snr_linear = 10**(snr_db / 10)
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
    return signal + noise

def calculate_ber(M, snr_db):
    """Calcula a BER simulada para um esquema PAM de ordem M."""
    num_symbols = 100000
    tx_symbols = generate_pam_symbols(M, num_symbols)
    rx_symbols = add_awgn_noise(tx_symbols, snr_db)

    # Demodulação (decisão por limiares)
    decision_levels = np.linspace(-M + 1, M - 1, M - 1) / np.sqrt(np.mean((np.arange(M) - (M - 1) / 2)**2))
    decisions = np.digitize(rx_symbols, decision_levels)  # Mapeia para índices
    decoded_symbols = 2 * decisions - (M - 1)
    decoded_symbols = decoded_symbols / np.sqrt(np.mean(decoded_symbols**2))

    # Calcula BER
    bit_errors = np.mean(tx_symbols != decoded_symbols)
    return bit_errors

def theoretical_ber(M, snr_db):
    """Calcula a BER teórica para PAM de ordem M sob AWGN."""
    snr_linear = 10**(snr_db / 10)
    return 2 * (1 - 1 / M) * (1 / np.log2(M)) * np.sqrt(3 / (2 * (M**2 - 1) * snr_linear))

def plot_ber_curves():
    """Plota curvas BER simuladas e teóricas para diferentes ordens de modulação PAM."""
    snr_db_range = np.arange(0, 35, 2)  # Faixa de SNR em dB
    M_values = [2, 4, 8, 16]  # Ordens de modulação PAM

    plt.figure(figsize=(10, 6))
    for M in M_values:
        simulated_ber = [calculate_ber(M, snr_db) for snr_db in snr_db_range]
        theoretical_ber_values = [theoretical_ber(M, snr_db) for snr_db in snr_db_range]

        # Adiciona ao gráfico
        plt.semilogy(snr_db_range, simulated_ber, marker='o', linestyle='-', label=f"Simulado {M}-PAM")
        plt.semilogy(snr_db_range, theoretical_ber_values, marker='x', linestyle='--', label=f"Teórico {M}-PAM")

    # Configurações do gráfico
    plt.title("Curvas BER para Modulação PAM", fontsize=14)
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Taxa de Erro de Bit (BER)", fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.ylim(1e-5, 1)
    plt.xlim(0, 35)
    plt.show()

# Chamada da função para plotar as curvas
plot_ber_curves()
