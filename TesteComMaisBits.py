import numpy as np
import matplotlib.pyplot as plt

# Configurações globais
num_symbols = 5000           # Número de símbolos
num_carriers = 64            # Subportadoras para OFDM
snr_levels = np.arange(0, 30, 2)  # Valores de SNR para análise
modulation_orders = [2, 4, 8, 16] # Ordens de modulação PAM

# Função para gerar símbolos PAM
def generate_pam_symbols(order, num_symbols):
    levels = np.arange(-(order-1), order, 2)  # Exemplo: [-3, -1, 1, 3] para 4-PAM
    bits_per_symbol = int(np.log2(order))
    data_bits = np.random.randint(0, order, size=num_symbols)
    symbols = levels[data_bits]
    return symbols, data_bits

# Geração de Símbolos Aleatórios para OFDM (16-QAM: 4 bits por subportadora):
qam_bits = np.random.randint(0, 16, size=(num_symbols, num_carriers))  # 4 bits/símbolo
qam_symbols = (2 * (qam_bits // 4) - 3) + 1j * (2 * (qam_bits % 4) - 3)  # Mapeamento QAM

# Modulação OFDM:
ofdm_signal = np.fft.ifft(qam_symbols, axis=1).flatten()

# Função para simular ruído AWGN
def add_awgn_noise(signal, snr_dB):
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_dB / 10))
    noise = np.random.normal(scale=np.sqrt(noise_power), size=signal.shape)
    return signal + noise

# Função de demodulação PAM
def pam_demodulate(received_signal, levels):
    decisions = np.argmin(np.abs(received_signal[:, None] - levels), axis=1)
    return decisions

# Função para calcular BER
def calculate_ber(original_bits, decoded_bits):
    errors = np.sum(original_bits != decoded_bits)
    ber = errors / len(original_bits)
    return ber

# Simulação de diferentes ordens de PAM
ber_results = {}

for order in modulation_orders:
    print(f"Simulando para {order}-PAM")
    pam_levels = np.arange(-(order-1), order, 2)
    bits_per_symbol = int(np.log2(order))
    
    ber_per_snr = []
    
    for snr in snr_levels:
        pam_symbols, original_bits = generate_pam_symbols(order, num_symbols)
        pam_signal = pam_symbols  # Sem pulse shaping neste exemplo

        # Adicionar ruído
        noisy_signal = add_awgn_noise(pam_signal, snr)

        # Demodulação
        decoded_bits = pam_demodulate(noisy_signal, pam_levels)

        # Calcular BER
        ber = calculate_ber(original_bits, decoded_bits)
        ber_per_snr.append(ber)

    ber_results[order] = ber_per_snr

ofdm_bers = []

for snr in snr_levels:
    # Adicionando ruído
    ofdm_noisy_signal = add_awgn_noise(ofdm_signal, snr)

    # Demodulação OFDM (16-QAM):
    ofdm_noisy_symbols = np.fft.fft(ofdm_noisy_signal.reshape((num_symbols, num_carriers)), axis=1)
    ofdm_decoded = np.round((ofdm_noisy_symbols.real + 3) / 2).astype(int).clip(0, 3) * 4 + \
                   np.round((ofdm_noisy_symbols.imag + 3) / 2).astype(int).clip(0, 3)

    # Cálculo do BER
    ofdm_ber = calculate_ber(qam_bits.flatten(), ofdm_decoded.flatten())

    ofdm_bers.append(ofdm_ber)

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
if len(snr_levels) == len(ofdm_bers):
    plt.figure(figsize=(10, 6))
    plt.plot(snr_levels, ofdm_bers, 'o-', label='OFDM BER')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.title('BER vs SNR')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
else:
    print("Erro: snr_levels e os valores de BER não têm o mesmo comprimento.")


plt.figure(figsize=(10, 6))
for order, ber_values in ber_results.items():
    plt.plot(snr_levels, ber_values,'o-', label=f"{order}-PAM")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.yscale('log')
plt.title("Comparativo de BER para diferentes ordens de PAM")
plt.legend()
plt.tight_layout()
plt.show()
