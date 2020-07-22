# -*- coding: utf-8 -*-
"""
Created on 07/20/2020

@author: Fan Fei

Version 07.20.2020: Full Hamiltonian and Fock State Construction
Version 07.21.2020: Fixed fermion sign problem. Add solve() method

"""

import numpy as np
import matplotlib.pyplot as plt

#fermi_dirac distribution
def fermi_dirac(E, T):
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
        


def test_1():
    """
    N = 100
    T = 0.1 
    x = np.linspace(-10,10,N)
    plt.plot(x, fermi_dirac(x,T))
    plt.show()
    """

    
class DotArray():
    def __init__(self, hsingle, coulomb, nsite, tL, tR):
        self.hsingle = hsingle
        self.coulomb = coulomb
        self.nsite = nsite
        self.nstate = 2**nsite
        self.temp = 0.1
        self.mu_L = 0.0
        self.mu_R = 0.0
        self.states = []
        self.p_vector = np.zeros(self.nstate)
        self.charge = np.zeros(self.nstate)
        self.tLeft = tL
        self.tRight = tR

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
        self.mL = np.zeros((self.nstate, self.nstate))
        self.mR = np.zeros((self.nstate, self.nstate))
        for i in range(0, self.nstate):
            for j in range(0, self.nstate):
                vec_a = self.eigen_vec[i]
                vec_b = self.eigen_vec[j]
                for dot in self.tLeft:
                    m_abj = self.product(vec_a, self.c_dagger(dot,vec_b))
                    self.mL[i,j] += m_abj**2
                for dot in self.tRight:
                    m_abj = self.product(vec_a, self.c(dot, vec_b))
                    self.mR[i,j] += m_abj**2

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
            self.p_vector[i] = np.exp(-(self.eigen_val[i]-self.charge[i]*self.mu_L)/self.temp)
        summ = sum(self.p_vector)
        self.p_vector = self.p_vector / summ




    def mu_leads(self, mu_L, mu_R):
        self.mu_L = mu_L
        self.mu_R = mu_R

    def charge_block(self):
        tmp = np.zeros((self.nstate, self.nstate))
        for i in range(0, self.nstate):
            tmp[i,i] = count_charge(i)
        for i in range(0, self.nstate):
            self.charge[i] = np.matmul(self.eigen_vec[i], np.matmul(tmp, self.eigen_vec[i]))



    def diagonize(self):
        self.eigen_val, self.eigen_vec = np.linalg.eigh(self.full_ham)
        self.eigen_vec = np.transpose(self.eigen_vec)

    def print_states(self):
        print("gg")

    def print_full_ham(self):
        print(self.full_ham)

    def solve(self):
        self.construct_states()
        self.construct_full_ham()
        self.diagonize()
        self.charge_block()
        self.construct_matrix()
        self.cal_p_factor()
        self.cal_effective_gamma()



if __name__ == "__main__":

    """
    x = np.linspace(-2,4,100)
    y = np.zeros(100)
    for i in range(0,100):
        hsingle = {(0,0):x[i]}
        s = DotArray(hsingle,0,1)
        if x[i] >= 0:
            s.charge = [0,1]
        else:
            s.charge = [1,0]
        s.construct_states()
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


    
    #####################################simple testing do not delete
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

    
    n = 1
    hsingle = {(0,0):2, (1,1):3, (0,1): 0.2}
    tL = [0]
    tR = [1]
    s = DotArray(hsingle,0,2,tL,tR)
    s.mu_L = 2
    x = np.linspace(-5,5,100)
    y = np.linspace(-5,5,100)
    z = np.zeros(100)
    stab = np.zeros((100,100))
    for i in range(0,100):
        print(i)
        for j in range(0,100):
            hsingle = {(0,0):x[i],(1,1):x[i]+1, (0,1): 1}
            s.change(hsingle=hsingle)
            s.mu_L = y[j]
            s.solve()
            stab[i,j] = abs(s.effective_gamma)

    stab_plot(stab, x, y)
    


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
    test_vec = [0,0,0,0,1,0,0.5,0]
    print(test_vec)
    result = s.c(1, test_vec)
    print(result)
    #####################################
    """