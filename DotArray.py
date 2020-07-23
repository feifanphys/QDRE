# -*- coding: utf-8 -*-
"""
Created on 07/20/2020

@author: Fan Fei

Version 07.20.2020: Full Hamiltonian and Fock State Construction
Version 07.21.2020: Fixed fermion sign problem. Add solve() method
Version 07.22.2020: Added e-e coulomb interaction. Ignoring modulated hopping term currently.

"""

import numpy as np
import matplotlib.pyplot as plt
import time

#fermi_dirac distribution
def fermi_dirac(E, T):
    if E/T > 10:
        return 0
    p = 1/(1+np.exp(E/T))
    return p


#return ab/(a+b)
def geometry(a,b):
    return a*b/(a+b)

def count_charge(n): 
    count = 0
    while (n): 
        count += n & 1
        n >>= 1
    return count

def count_fermion(n, stop):
    count = 0
    copy = n
    i = 0
    while (i < stop): 
        count += copy & 1
        copy >>= 1
        i += 1
    return count

def count_fermion_r(n, start):
    count = 0
    i = 0
    while (i < start):
        n >>= 1
        i += 1
    while(n):
        count += n & 1
        n >>= 1
    return count
        


    
class DotArray():
    def __init__(self, hsingle, coulomb, nsite, tL, tR):
        self.hsingle = hsingle
        self.coulomb = coulomb
        self.nsite = nsite
        self.nstate = 2**nsite
        self.temp = 0.6
        self.mu_L = 0.0
        self.mu_R = 0.0
        self.states = []
        self.p_vector = np.zeros(self.nstate)
        self.charge = np.zeros(self.nstate)
        self.tLeft = tL
        self.tRight = tR
        self.mL = np.zeros((self.nstate, self.nstate))
        self.mR = np.zeros((self.nstate, self.nstate))

    def change(self, hsingle=None):
        self.hsingle = hsingle


    def construct_states(self):
        for i in range(0, self.nstate):
            binary = "{0:b}".format(i)
            if len(binary) < self.nsite:
                binary = "0"* (self.nsite - len(binary)) + binary
            state = {"id": i, "str": binary}
            self.states.append(state)
    
    def print_fock_state(self):
        for i in range(0, self.nstate):
            print(str(self.states[i]["id"]) + ":   |" + self.states[i]["str"] + ">")

    def construct_full_ham(self):
        #start = time.time()
        self.full_ham = np.zeros((self.nstate,self.nstate))
        for entry in self.hsingle:
            dot_1, dot_2 = entry[0], entry[1]
            pairs = self.hij(dot_1, dot_2)
            for pair in pairs:
                if dot_1 == dot_2:
                    self.full_ham[pair[0],pair[1]] += self.hsingle[entry]
                else:
                    #fermionic sign
                    fsign = np.power(-1, count_fermion(pair[0], dot_1) + count_fermion(pair[1], dot_2)) * (+1 if dot_1 < dot_2 else -1)
                    #print(pair[0], pair[1])
                    #print(fsign)
                    self.full_ham[pair[0],pair[1]] += self.hsingle[entry] * fsign
                    self.full_ham[pair[1],pair[0]] += self.hsingle[entry] * fsign
        #end = time.time() - start
        #print("constructing H time--- %s seconds ---" % (end))

        for entry in self.coulomb:
            #print(entry)
            dot_1, dot_2, dot_3, dot_4 = entry[0], entry[1], entry[2], entry[3]
            pairs = self.hjklm(dot_1, dot_2, dot_3, dot_4)
            for pair in pairs:
                ffsign = np.power(-1, count_fermion(pair[0], dot_3) + count_fermion(pair[1], dot_4)) * (+1 if dot_3 < dot_4 else -1)
                ffsign = ffsign * np.power(-1, count_fermion(pair[0], dot_2) + count_fermion(pair[1], dot_1)) * (+1 if dot_2 > dot_1 else -1)
                self.full_ham[pair[0],pair[1]] += self.coulomb[entry] * fsign


    def change_hsingle(self, hsingle):
        self.hsingle = hsingle

    def tleads(self, tL, tR):
        self.tLeft = tL
        self.tRight = tR

    def print_tleads(self):
        print(self.tLeft)
        print(self.tRight)

    #return all pair of states related by ij term in hsingle
    def hij(self,i,j):
        pairs = []
        for k in range(0,self.nsite):
            if k!=i and k!=j:
                #print(k)
                tmp = []
                for p in pairs:
                    #print(p)
                    tmp.append((p[0] + 2**k, p[1]+ 2**k))
                pairs.extend(tmp)
                pairs.append((2**i + 2**k, 2**j + 2**k))
                #print(pairs)
        pairs.append((2**i, 2**j))
        return pairs

    def hjklm(self, j, k, l, m):
        pairs = []
        pairs.append((2**j + 2**k, 2**l + 2**m))
        for dd in range(0, self.nsite):
            if dd not in [j,k,l,m]:
                tmp = []
                for p in pairs:
                    tmp.append((p[0] + 2**dd, p[1]+ 2**dd))
                pairs.extend(tmp)
        return pairs


    #return c_j |x> for input of any state |x>
    def c(self, j, input_state):
        vec = np.zeros(self.nstate)
        for i in range(len(input_state)):
            dummy = (i & 2**j)
            if dummy > 0:
                vec[i-dummy] = input_state[i]

        return vec

    #return c_j^dagger |x> for input of any state |x>
    def c_dagger(self, j, input_state):
        vec = np.zeros(self.nstate)
        for i in range(len(input_state)):
            dummy = (i & 2**j)
            if dummy == 0:
                vec[i+2**j] = input_state[i]

        return vec

    def product(self, vec_a, vec_b):
        return np.matmul(vec_a, vec_b)

    def amp(self, vec_a, op, vec_b):
        product = np.matmul(vec_a, np.matmul(op, vec_b))
        amplitude = product**2
        return amplitude

    def construct_matrix(self):
        #start = time.time()
        for i in range(0, self.nstate):
            for j in range(0, self.nstate):
                self.mL[i,j] = 0
                self.mR[i,j] = 0
                vec_a = self.eigen_vec[i]
                vec_b = self.eigen_vec[j]
                for dot in self.tLeft:
                    if self.charge[i] == (self.charge[j] + 1):
                        m_abj = np.matmul(vec_a, self.c_dagger(dot,vec_b))
                        self.mL[i,j] += m_abj**2
                for dot in self.tRight:
                    if self.charge[i] == (self.charge[j] - 1):
                        m_abj = np.matmul(vec_a, self.c(dot,vec_b))
                        self.mR[i,j] += m_abj**2
        #end = time.time() - start
        #print("constructing M time--- %s seconds ---" % (end))

    def cal_effective_gamma(self):
        gamma = 0
        for i in range(0, self.nstate):
            for j in range(0, self.nstate):
                f_factor = 1 - fermi_dirac(self.eigen_val[i]-self.eigen_val[j]-self.mu_L, self.temp)
                if self.mL[i,j] != 0:
                    gamma += self.mL[i,j] * self.mR[j,i] * f_factor * self.p_vector[i] / (self.mL[i,j] + self.mR[j,i]) 
                #print(f_factor)
        self.effective_gamma = gamma

    def cal_p_factor(self):
        for i in range(0, self.nstate):
            exponential = -(self.eigen_val[i]-self.charge[i]*self.mu_L)/self.temp
            self.p_vector[i] = np.exp(exponential)
        summ = sum(self.p_vector)
        self.p_vector = self.p_vector / summ




    def mu_leads(self, mu_L, mu_R):
        self.mu_L = mu_L
        self.mu_R = mu_R

    def charge_block(self):
        #start = time.time()
        tmp = np.zeros((self.nstate, self.nstate))
        for i in range(0, self.nstate):
            tmp[i,i] = count_charge(i)
        for i in range(0, self.nstate):
            self.charge[i] = round(np.matmul(self.eigen_vec[i], np.matmul(tmp, self.eigen_vec[i])))
        #end = time.time() - start
        #print("charge block time--- %s seconds ---" % (end))



    def diagonize(self):
        #start = time.time()
        self.eigen_val, self.eigen_vec = np.linalg.eigh(self.full_ham)
        self.eigen_vec = np.transpose(self.eigen_vec)
        #end = time.time() - start
        #print("diagonizing time--- %s seconds ---" % (end))

    def print_states(self):
        print("gg")

    def print_full_ham(self):
        print(self.full_ham)

    def solve(self):
        #CPU expensive
        self.construct_full_ham()
        self.diagonize()
        self.charge_block()
        self.construct_matrix()

        #CPU non-expensive
        self.cal_p_factor()
        self.cal_effective_gamma()



if __name__ == "__main__":

    """
    ###Line scan

    x = np.linspace(-2,4,100)
    y = np.zeros(100)
    for i in range(0,100):
        hsingle = {(0,0):x[i]}
        s = DotArray(hsingle,0,1)
        s.construct_full_ham()
        s.diagonize()
        tL = [0]
        tR = [0]
        s.tleads(tL, tR)
        s.construct_matrix()
        s.cal_p_factor()
        s.cal_effective_gamma()
        y[i] = s.effective_gamma
    plt.plot(x,y)
    plt.show()
    """

    """
    ### diagonize check
    hsingle = {(0,0):2, (1,1):3, (0,1): 0.5, (2,2):4, (3,3):5 , (0,2): 0.6, (1,2): 0.7, (2,3): 0.5, (1,3):0.3, (4,4): 3, (3,4):0.2, (0,4): 0.1, (5,5): 7, (3,5): 0.3,
                (6,6): 5, (7,7):4, (2,7): 0.4, (4,6): 0.2}
    coulomb = {(0,1,1,0): 0.1,
                (1,2,2,1): 0.2,
                (2,3,3,2): 0.2,
                (3,4,4,3): 0.2,
                (1,5,5,1): 0.1}
    tL = [0]
    tR = [1]
    s = DotArray(hsingle, coulomb, 8, tL, tR)
    start_time = time.time()
    s.solve()
    total_time = time.time() - start_time
    #print(s.full_ham)
    #print(s.charge)
    #print(s.eigen_val)
    print("total--- %s seconds ---" % (total_time))
    """

    
    #####################################stab_plot
    def stab_plot(stab, vlst, vglst):
        (xmin, xmax, ymin, ymax) = np.array([vglst[0], vglst[-1], vlst[0], vlst[-1]])
        fig = plt.figure(figsize=(8,6))

        p1 = plt.subplot(1, 1, 1)
        p1.set_xlabel('$E_1}(mV)$', fontsize=20)
        p1.set_ylabel('$\mu_L(mV)$', fontsize=20)
        p1_im = plt.imshow(stab.T, extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower', cmap = plt.get_cmap('Spectral'))
        cbar1 = plt.colorbar(p1_im)
        cbar1.set_label('Current [$\Gamma$]', fontsize=20)
        plt.tight_layout()
        plt.show()

    """
    hsingle = {(0,0):2, (1,1):3, (0,1): 0.2}
    coulomb = {}
    tL = [0]
    tR = [1]
    s = DotArray(hsingle, coulomb, 2, tL, tR)

    steps = 100
    x = np.linspace(-5,5,steps)
    y = np.linspace(-5,5,steps)
    stab = np.zeros((steps,steps))
    start_time = time.time()
    for i in range(0,steps):
        print(i)
        for j in range(0,steps):
            hsingle = {(0,0):x[i],(1,1):x[i]+1, (0,1): 1}
            s.change(hsingle=hsingle)
            s.mu_L = y[j]
            s.solve()
            stab[i,j] = abs(s.effective_gamma)
    total_time = time.time() - start_time
    print("--- %s seconds ---" % (total_time))
    print("--- %s seconds per point---" % (total_time/(steps**2)))
    stab_plot(stab, x, y)
    """


    #print(s.effective_gamma)
    #s.construct_states()
    #print(s.hij(1,2))

    #s.construct_full_ham()
    #print("Full Hamiltonian is:")
    #s.print_full_ham()
    #s.diagonize()
    #s.charge_block()
    #print("charge block is: ")
    #print(s.charge)
    #print("Eigenvalues and eigenvecs:")
    #print(s.eigen_val)
    #print(s.eigen_vec)
    #tL = [0]
    #tR = [0]
    #s.tleads(tL, tR)
    #print("tLeads:")
    #s.print_tleads()
    #s.construct_matrix()
    #print("Matrix elements:")
    #print(s.mL)
    #print(s.mR)

    #s.cal_p_factor()
    #print("p factor is:")
    #print(s.p_vector)

    #s.cal_effective_gamma()
    #print("Effective gamma is: ")
    #print(s.effective_gamma)
    """
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

    steps = 100
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
    """