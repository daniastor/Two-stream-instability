"""
Two Stream Instability - Electron Holes
Code TSIEH

Author:
    Daniela F. López Astorquiza
    Jaime Humberto Hoyos Barrios
"""
#    Main Python Libraries
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
matplotlib.rcParams.update({'font.size': 30})

#    Parameters
N  = 15000 #Particles number
L  = 10    #Debye lengths
vt = 1.0   #Thermal velocity
vb = 3.0   #Average velocity
ng = 3750  #Grid points umber
dx = L/ng  #Space step
dt = 0.05  #Time step
nt = 2#250   #Iterations number
Q = -L/N   #Super particle charge
k = np.fft.fftfreq(ng, dx) #Frequencies
k[0] = 1.0                 #Fix frequencies

#    Initial velocity and position distribution
x = np.arange(N)*(L/N)
v = vt*np.random.randn(N)
pm = 2*np.random.randint(0,2,N) - 1
v = v+pm*vb

Xinicial = x
Vinicial = v

for t in range(nt):
    k1 = v*dt
    k2 = (v+(1/2)*k1)*dt
    k3 = (v+(1/2)*k2)*dt
    k4 = (v+k3)*dt
    x = x + (1/6)*(k1+2*k2+2*k3+k4)
    x = np.mod(x, L)
    
    P = x/dx
    PG = np.floor(P)
    w = 1.0 + PG - P
    
    G = np.int32(np.mod(np.concatenate([PG,PG+2.5]),ng))
    WW = np.concatenate([w,2.5-w])
    
    rho = np.zeros(ng)
    for G1,W1 in zip(G,WW):
        rho[G1] = rho[G1]+W1
    rho = Q/dx*(rho)
    
    rhok = np.fft.fft(rho) #Density, frequency domain
    phi= np.fft.ifft((rhok/k**2)).real
    phi1 = phi[G]
    phi2= (phi1[:N]+phi1[N:]) #Electric Potential
    ee = np.fft.ifft(-1j*rhok/k).real
    ep = ee[G]
    E = (ep[:N] + ep[N:]) #Electric field
    
    l1 = (-E)*dt
    l2 = (-E+(1/2)*l1)*dt
    l3 = (-E+(1/2)*l2)*dt
    l4 = (-E+l3)*dt
    v = v + (1/6)*(l1+2*l2+2*l3+l4)
    print(t)
    
    #    Generation of graphics
    fig=plt.figure(figsize=(20, 13), dpi=80)
    plt.subplot(2, 2, 1)
    plt.plot(x,v,".",alpha=0.4) #Phase space
    plt.ylim(-8,8)
    plt.xlim(0,L)
    plt.xlabel("x/\u03BBD")
    plt.ylabel("v/(\u03BBD\u03C9p)")
    plt.tick_params(labelsize=30)
    plt.text(9,6.3,'(a)')
    
    plt.subplot(2, 2, 3)
    plt.plot(x,phi2,"k.") #Electric Potential
    plt.ylim(-70,80)
    plt.xlim(0,L)
    plt.xlabel("x/\u03BBD")
    plt.ylabel("\u03A6(x)e/(kTe)")
    plt.tick_params(labelsize=30)
    plt.text(9,63.3,'(c)')
    
    plt.subplot(2, 2, 2)
    plt.plot(x,E,"k.") #Electric field
    plt.ylim(-19.5,19.5)
    plt.xlim(0,L)
    plt.xlabel("x/\u03BBD")
    plt.ylabel("E(x)e\u03BBD/(kTe)")
    plt.tick_params(labelsize=30)
    plt.text(9,15.5,'(b)')
    
    m=[i for i,x1 in enumerate(x) if (x1>=4.5) and (x1<=5.5)]
    a0=v[m]
    plt.subplot(2, 2, 4)
    valores0 = sns.kdeplot(a0,vertical=True,color="k",visible=True)
    plt.xlim(0,0.2)
    plt.ylim(-9.5,9.5)
    plt.xlabel("f($v_{x}$)")
    plt.ylabel("v/(\u03BBD\u03C9p)")
    plt.tick_params(labelsize=30)
    plt.text(0.18,7.8,'(d)')
    fig.savefig("TSIEH_{}.png".format(t+1), transparent=False)
    plt.close()
    
#    Energy contours last Iteration
for save_way in range(2):
    fig2 = plt.figure(figsize=(20, 13), dpi=80)
    et = (1/2)*v**2-phi2 #Energies
    D = 700
    res = et[1::D]
    print("Energies: "+str(res))
    v1 = np.zeros(len(phi2))
    v2 = np.zeros(len(phi2))
    for l in range(len(res)):
        B = res[l]
        for o in range(len(phi2)):
            v1[o]=np.sqrt(B+phi2[o])*2
            v2[o]=-np.sqrt(B+phi2[o])*2
        if (save_way==0):
            fig3 = plt.figure(figsize=(20, 13), dpi=80)
        plt.plot(x,v1,"k.")
        plt.plot(x,v2,"k.")
        plt.xlim(0,L)
        plt.ylim(-22,22)
        plt.xlabel("x/\u03BBD", fontsize=60)
        plt.ylabel("v/(\u03BBD\u03C9p)",fontsize=60)
        plt.tick_params(labelsize=60)
        if (save_way==0): #Save contours separately
            plt.title("Energy contour with \u03B5 = "+str(B.round(2)))
            fig3.savefig('Energy_contour_{}.png'.format(l+1))
            plt.close()
    #Now save contours together
    plt.title("Energy contours")
    fig2.savefig('Energy_contours.png', transparent=False)
    plt.close()

#    Distribution functions last Iteration
c_all = []
b_all = []
z_all = []
for j1 in range(0,L):
    fig4 = plt.figure(figsize=(25, 15), dpi=80)
    m=[i for i,x1 in enumerate(x) if (x1>j1+0.5) and (x1<=j1+1.5)]
    a=v[m]
    valores = sns.kdeplot(a,vertical=True,color="k",visible=True)
    c,b = valores.get_lines()[0].get_data()
    c_all.append(c)
    b_all.append(b)
    zz =  [[j1+1 for x in range(200)] for x in range(1)]
    z_all.append(zz)
    plt.title("Distribution Function, around x="+str(j1+1))
    plt.xlim(0,0.2)
    plt.ylim(-9.5,9.5)
    plt.xlabel("f($v_{x}$)", fontsize=60)
    plt.ylabel("v/(\u03BBD\u03C9p)",fontsize=60)
    plt.tick_params(labelsize=60)
    fig4.savefig('Distribution_Function_around_{}.png'.format(j1+1))
    plt.close()

# Three-dimensional representation
c_all = np.vstack(c_all)
b_all = np.vstack(b_all)
z_all = np.vstack(z_all)
fig5 = plt.figure(figsize=(25, 20), dpi=80)
ax = plt.axes(projection='3d')
plt.tick_params(labelsize=60)
ax.tick_params(axis='x', which='major',pad=10)
ax.set_xlabel("v",fontsize=60,labelpad=50)
ax.tick_params(axis='y', which='major', pad=10)
ax.set_ylabel("x",fontsize=60,labelpad=50)
ax.tick_params(axis='z', which='major', pad=40)
ax.set_zlabel("f($v_{x}$)",fontsize=60,labelpad=110)
Hist = ax.plot_surface(b_all,z_all,c_all,cmap='plasma') #Distribution Function
cbar = plt.colorbar(Hist, pad=-0.05,shrink=0.8)
cbar.ax.tick_params(labelsize=60)
ax.view_init(60, 80)
fig5.savefig('All_Distribution_Function.png', transparent=False)
#