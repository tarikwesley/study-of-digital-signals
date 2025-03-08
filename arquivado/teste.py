import numpy as np
import matplotlib.pyplot as plt

# Dados fictícios para o exemplo
snr_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40]
pam_bers = [0.2, 0.15, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 1e-4]
ofdm_bers = [0.15, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 1e-4, 1e-5]

plt.figure(figsize=(12, 8))

# Gráfico de linhas com anotações
plt.plot(snr_levels, pam_bers, 'o-', color='blue', label='PAM BER')
plt.plot(snr_levels, ofdm_bers, 's-', color='green', label='OFDM BER')

# Adicionar anotações nos pontos de interesse
for i, snr in enumerate(snr_levels):
    plt.annotate(f'{pam_bers[i]:.1e}', (snr, pam_bers[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    plt.annotate(f'{ofdm_bers[i]:.1e}', (snr, ofdm_bers[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

# Adicionar sombreamento
plt.fill_between(snr_levels, pam_bers, ofdm_bers, color='gray', alpha=0.2, label='Diferença PAM-OFDM')

plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.yscale('log')
plt.title('BER vs SNR para PAM e OFDM com Anotações')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
