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

cL1,cG1,cm = 1.6, 0.6, 1.0
cL2,c12,c21,cG2 = 0.3, 0.2, 0.2, 0.5
cR2,cR1 = 1.6, 0.3

q10 = 0.0 
q20 = 0.0

csigma1 = cL1 + cG1 + cm + c12 + cR1
csigma2 = cR2 + cG2 + cm + c21 + cL2

U1 = (csigma2/(csigma1*csigma2-cm**2))/6.24*1000
U2 = (csigma1/(csigma1*csigma2-cm**2))/6.24*1000
Um = (cm/(csigma1*csigma2 - cm**2))/6.24*1000

vgate1, vgate2, vbiasL, vbiasR = 0.0, 0.0, 0.0, 0.0

q1 = (c12*vgate2 + cG1*vgate1 + cL1*vbiasL + cR1*vbiasR)*6.24/1000 + q10
q2 = (c21*vgate1 + cG2*vgate2 + cL2*vbiasL + cR2*vbiasR)*6.24/1000 + q10

mu1 = Um*q2 + U1*q1 - U1/2
mu2 = Um*q1 + U2*q2 - U2/2

Eq1 = 0.5 * U1
Eq2 = 0.0 * U1
Eq1h = 0.5 * U1
Eq2h = 0.0 * U1

omegapres, omegaflip = 0.4 , 0.0

hsingle =  {(0,0): Eq1-mu1,
            (1,1): Eq1-mu1,
            (2,2): Eq2-mu2,
            (3,3): Eq2-mu2,
            (0,2): -omegapres,
            (1,3): -omegapres,
            (0,3): -omegaflip,
            (1,2): -omegaflip
}

coulomb = {(0,1,1,0):U1,
          (1,2,2,1):Um,
          (0,2,2,0):Um,
          (1,3,3,1):Um,
          (0,3,3,0):Um,
          (2,3,3,2):U2
}




tL = [0,1]
tR = [2,3]
s = DotArray(hsingle, coulomb, 4, tL, tR)

steps = 100
x = np.linspace(-100,100,steps)
y = np.linspace(200,400,steps)
stab = np.zeros((steps,steps))
start_time = time.time()

for i in range(0,steps):
    print(i)
    vgate1 = y[i]
    for j in range(0,steps):
        vgate2 = x[j]

        q1 = (c12*vgate2 + cG1*vgate1 + cL1*vbiasL + cR1*vbiasR)*6.24/1000 + q10
        q2 = (c21*vgate1 + cG2*vgate2 + cL2*vbiasL + cR2*vbiasR)*6.24/1000 + q20

        mu1 = Um*q2 + U1*q1 - U1/2
        mu2 = Um*q1 + U2*q2 - U2/2

        hsingle =  {(0,0): Eq1-mu1,
            (1,1): Eq1-mu1,
            (2,2): Eq2-mu2,
            (3,3): Eq2-mu2,
            (0,2): -omegapres,
            (1,3): -omegapres,
            (0,3): -omegaflip,
            (1,2): -omegaflip}

        s.change(hsingle=hsingle)
        s.solve()
        stab[i,j] = abs(s.effective_gamma)

total_time = time.time() - start_time
print("--- %s seconds ---" % (total_time))
print("--- %s seconds per point---" % (total_time/(steps**2)))
stab_plot(stab, x, y)