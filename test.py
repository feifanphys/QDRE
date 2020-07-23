import numpy as np
import matplotlib.pyplot as plt
import time
from DotArray import DotArray


def stab_plot(stab, vlst, vglst):
    (xmin, xmax, ymin, ymax) = np.array([vglst[0], vglst[-1], vlst[0], vlst[-1]])
    fig = plt.figure(figsize=(8,6))

    p1 = plt.subplot(1, 1, 1)
    p1.set_xlabel('$V_{g1}(mV)$', fontsize=20)
    p1.set_ylabel('$V_{g2}(mV)$', fontsize=20)
    p1_im = plt.imshow(stab.T, extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower', cmap = plt.get_cmap('Spectral'))
    cbar1 = plt.colorbar(p1_im)
    cbar1.set_label('Current [$\Gamma$]', fontsize=20)
    plt.tight_layout()
    plt.show()

Cg1 = 2.6
Cg2 = 2.4
Cs = 2.8
Cd = 2.8
Csigma = Cg1 + Cg2 + Cs + Cd

E_c = (1/Csigma)/6.24*1000

Q_0 = 0

hsingle = {(0, 0): 0.5*E_c - 3 * E_c,
        (1, 1): 0.5*E_c - 2 * E_c,
        (2, 2): 0.5*E_c - 1 * E_c,
        (3, 3): 0.5*E_c + 0 * E_c}

coulomb = {}
tL = [0,1,2,3]
tR = [0,1,2,3]
s = DotArray(hsingle, coulomb, 4, tL, tR)

steps = 50
x = np.linspace(-200,0,steps)
y = np.linspace(0,200,steps)
stab = np.zeros((steps,steps))
start_time = time.time()

for i in range(0,steps):
    print(i)
    for j in range(0,steps):
        xxx = - Cg2/Csigma * x[j] - Cg1/Csigma * y[i]
        hsingle = {(0, 0): 0.5*E_c - 3 * E_c + xxx,
                (1, 1): 0.5*E_c - 2 * E_c + xxx,
                (2, 2): 0.5*E_c - 1 * E_c + xxx,
                (3, 3): 0.5*E_c + 0 * E_c + xxx}
        s.change(hsingle=hsingle)
        s.solve()
        stab[i,j] = abs(s.effective_gamma)

total_time = time.time() - start_time
print("--- %s seconds ---" % (total_time))
print("--- %s seconds per point---" % (total_time/(steps**2)))
stab_plot(stab, x, y)