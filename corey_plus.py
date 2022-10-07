import numpy as np 
import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
import math
import os
import argparse
from moviepy.editor import *

####################################################################
#
# Flux function
#
####################################################################

def flux_basic(S):   
    global a, b, c, mu_w, mu_g, mu_o
    
    s_w = S[0]; s_g = S[1]
    
    # caption and projection of outliers
    if s_w < 0:
        s_w = 0
    if s_g < 0:
        s_g = 0
    if s_w + s_g > 1:
        s_norm = (s_w**2 + s_g**2)**0.5
        s_w = s_w / s_norm
        s_g = s_g / s_norm
    
    lambda_w = s_w**a / mu_w
    lambda_g = s_g**b / mu_g
    s_o = 1 - s_w - s_g
    lambda_o = s_o**c / mu_o
    lambda_total = lambda_w + lambda_g + lambda_o
    f_w = lambda_w / lambda_total 
    f_g = lambda_g / lambda_total

    F=np.array([f_w, f_g])
    return F

####################################################################
#
# Test flux function
#
####################################################################

def test_flux():
    S = np.array([0.1, 0.2])
    F = flux_basic(S)
    F = flux_improved(S)
    print('S:', S)
    print('F:', F)
    
#########################################################################################################
# >    
    

####################################################################
#
# Flux function - improved
#
####################################################################

# The flux function for Corey/Stone I model
def flux_improved(S):
    global a, b, c, mu_w, mu_g, mu_o
    sw = S[0]; sg = S[1]
    
    
    """
      The flow is quite flexible, allowing to implement permeabilities 
      of the convex Corey type or the Stone I type, which is generalized 
      by the possibility of placing power laws different from the 
      quadritric ones.
        The foam model can be incorporated by activating foamM as 'foam'. 
      Several functions are calculated inline, which makes flux_improved(S) 
      computation more efficient by eliminating many conditionals that 
      define the model in question.
    """
    # Flag for Corey or Stone permeability model
    model = 'Corey'       # Other can be 'Stone'
    foamM = 'foam'     # It can be possible to activate the foam
                          # model given in Stars for sw, sg.
        
    # # Corey quadratic power laws
    #nw  = 2; ng  = 2; no = 2
    # Some natural power laws
    nw  = 3.29; ng  = 2.65; no = 5.84
    # Stone I power laws, conventional Stone I uses ngo = nwo = 2.
    nwo = 2.00; ngo = 2.00
    # Mathematical viscosities for nice graphic resolution
    # muw = 1; mug = 0.5; muo = 2
    
    # Convex Corey permeability model (these are mobilities)
    mw   = lambda x: x**nw / mu_w
    dmw  = lambda x: nw*(x**(nw - 1)) / mu_w
    mg0  = lambda x: x**ng / mu_g
    dmg0 = lambda x: x*(sg**(ng - 1)) / mu_g
    if model == 'Corey':
        # Corey oil permeability depends upon oil saturation
        mo  = lambda sw, sg: (1 - sw - sg)**no / mu_o
        dmo = lambda sw, sg: no*((1 - sw - sg)**(no - 1)) / mu_o
        dmodsw = lambda sw, sg: -dmo(sw, sg)
        dmodsg = lambda sw, sg: -dmo(sw, sg)
    elif model == 'Stone':
        # For Stone I model, the oil permeability is different
        mo = lambda sw, sg: (1 - sw - sg)*(1 - sw)**(nwo - 1)*(1 - sg)**(ngo - 1) / mu_o 
        dmodsw = lambda sw, sg: -( (1 - sw)**(nwo - 1) + (nwo - 1)*(1 - sw - sg)*((1 - sw)**(nwo - 2)))*(1 - sg)**(ngo - 1) / mu_o
        dmodsg = lambda sw, sg: -( (1 - sg)**(ngo - 1) + (ngo - 1)*(1 - sw - sg)*((1 - sg)**(ngo - 2)))*(1 - sw)**(nwo - 1) / mu_o
            
    mTotal = lambda sw, sg: mw(sw) + mg0(sg) + mo(sw, sg)
    dmTdsw = lambda sw, sg: dmw(sw) + dmodsw(sw, sg)
    dmTdsg = lambda sw, sg: dmg0(sg) + dmodsg(sw, sg)
        
    """ When foam model is activated, some parameters need to be settled """
    # if foamM == 'foam':
    if True:        
        fmmob = 54000; fmdry = 0.316; epdry = 6000
        fmoil = 0.200; floil = 0.120; epoil = 4.00
        # From no-foam to strong foam via the limiting water saturation
        foam_pol = lambda x: ((0.1875*x**2 -0.625)*x**2 + 0.9375)*x + 0.5
        def F_2(sw):
            x = 2.*epdry*(sw - fmdry) 
            if x < -1:
                return 0.
            elif x > 1:
                return 1.
            else:
                return foam_pol(x)
        # Oil detrimetal effect on foam
        def F_3(so):
            if so < floil:
                return 1.
            elif so > fmoil:
                return 0.
            else:
                return ((fmoil - so)/(fmoil - floil))**epoil
        def dFRM(sw, sg):
            # In oil quality change?
            so = 1 - sw - sg
            if so < floil or so > fmoil:
                return 0., 0.
            # In water quality change?
            x  = 2.*epdry*(sw - fmdry)
            if abs(x) >= 1:
                return 0., 0.
            fmmobInv = 1/(1. + fmmob*F_2(sw)*F_3(1 - sw - sg))**2
            dF2 = 1.875*epdry*(x**2 - 1)**2
            dF3 = (((fmoil - so)/(fmoil - floil))**(epoil - 1) )/(floil - fmoil)
            dFRMdsw = -fmmob*dF2*F_3*fmmobInv
            dFRMdsg = -fmmob*F_2*dF3*fmmobInv
            return dFRMdsw, dFRMdsg
        mg  = lambda sw, sg: mg0(sg)/(1. + fmmob*F_2(sw)*F_3(1 - sw - sg))
        dmg = lambda sw, sg: dmg0(sg) #""" WARNING: this is not the correct one """
    else:
        mg  = lambda sw, sg: mg0(sg)
        dmg = lambda sw, sg: dmg0(sg)    
    
    
    
    """ The fluxes are calculated first """
    fw = lambda sw, sg: mw(sw) / mTotal(sw, sg)
    fg = lambda sw, sg: mg(sw, sg) / mTotal(sw, sg)  
    
    F = np.array([fw(sw, sg), fg(sw, sg)])
    return F




# The Jacobian for flux function for Corey/Stone I model
def Jacobian(S):
    sw = S[0]; sg = S[1]
    
    fww = dfwdsw(sw, sg)
    fwg = dfwdsg(sw, sg)
    fgw = dfgdsw(sw, sg)
    fgg = dfgdsg(sw, sg)
    
    J = np.array([[fww, fwg], [fgw, fgg]])
    return J


def Jacobian_test(S, h):
    Jexact = Jacobian(S)
    FN = flux(S + h*np.array([0, 1]))
    FS = flux(S - h*np.array([0, 1]))
    
    FE = flux(S + h*np.array([1, 0]))
    FW = flux(S - h*np.array([1, 0]))
    
    hinv2 = 1./(2.*h)
    Jnum = np.array([[(FE[0] - FW[0])*hinv2, (FN[0] - FS[0])*hinv2], [(FE[1] - FW[1])*hinv2, (FN[1] - FS[1])*hinv2]])
    
    return Jexact - Jnum, Jexact, Jnum



# <
#########################################################################################################

####################################################################
#
# Save video
#
####################################################################

import os
from moviepy.editor import *

def save_video(dir_path):
    # global dir_path 
    # folder path
    # os.listdir(".")
    global path_num
    print('save video')
    # list file and directories
    res = os.listdir(dir_path)
    files = sorted(res) 
    png_files=[]
    for file in files:
        if (file.endswith(".png")) and ('test' in file):
            png_files.append(file)
            
    os.chdir(dir_path)
    # print("Current working directory: {0}".format(os.getcwd()))
    fps = 24
    clip = ImageSequenceClip(png_files, fps = fps) # fast 
    # clip.write_videofile("video-"+ dir_path + "-24.mp4", fps = 24)
    clip.write_videofile("video-"+str(path_num)+".mp4", fps = 24)
    
    os.chdir('..')
    

####################################################################
#
# Initialize
#
####################################################################
# def initialize():    
def initialize(x_span, n_cells, t_span, dt):
    global dx
    print('init - dt:', dt)
    # for sequential numbering of figures printed to files

    # calculate interval length    
    print(x_span[1], x_span[0], n_cells)
    dx = ( x_span[1] - x_span[0] ) / n_cells
    
    # calculate location of cell centers
    x = np.linspace(x_span[0] + dx/2, x_span[1] - dx/2, n_cells )
  
    # the solution vector is contained in the variable S
    # 2 columns and 'n_cell' rows    
    S = np.zeros((2, n_cells))

    # define the initial condition
    for k in range(0, n_cells):
        S[0, k] = k
        S[1, k] = 100+k
        if (k % 20) < 15:
            S[0, k] = 0.1
            S[1, k] = 0.2
        else:
            S[0, k] = 0.2
            S[1, k] = 0.3
        
        S[0, k] = 0; S[1, k] = 0 # oil
        # S[0, k] = 0; S[1, k] = 1 # gas # ERR with flux_improved        
        # S[0, k] = 0; S[1, k] = 0.95 # gas
        # S[0, k] = 0; S[1, k] = 0.98 # gas
        # S[0, k] = 0; S[1, k] = 0.995 # gas
        # S[0, k] = 0; S[1, k] = 0.999 # gas
        
    # S= np.zeros((2, n_cells)) # oil
        
    # Calculate the time step "dt" that is optimal for stability
    # CFL = 0.9;
    CFL = 1;
    amax = 0.1
    amax = 0.2
    # amax = 1
    # amax = 0.01
    #
    # dt = CFL * dx/abs(amax)
    
    # time instances of simulation
    T = np.arange (t_span[0], t_span[1], dt)
       
    return x, T, S  
    
####################################################################
#
# Vizualize the solution
#
####################################################################
    
def triangle(sw, sg):
    """ This rutine transforms orthogonal coordinates [sw, sg] into 
        baricentric coordinates as [x, y] in the plane. G is at the
        origin, W at rigth (1, 0) and O at top (0.5, 0.8660).
    """
    theta      = np.pi/3
    rotation   = np.array([[np.cos(theta), -np.cos(theta)],
                            [-np.sin(theta), -np.sin(theta)]])
    traslation = np.array([np.cos(theta), np.sin(theta)]).reshape((2, 1))
    
    S = np.empty([2, len(sw)])
    S[0, :] = sw
    S[1, :] = sg
    return rotation@[sw, sg] + traslation@np.ones([1, len(sw)])    
    

def plot_triangle(S):    
    # Triangle variant 1
    # These are the vertecis of the baricentric triangle
    W = triangle([1], [0])
    G = triangle([0], [1])
    O = triangle([0], [0])
    
    # The localization of which compound the triangle.
    plt.plot([W[0], G[0], O[0], W[0]], [W[1], G[1], O[1], W[1]])
    # plt.show()
    
    # plot triangle variant 2
    plt.plot([0, 1, 0.5, 0], [0, 0, math.sqrt(3)/2, 0], color='grey', marker='o', linestyle='solid', linewidth=5, markersize=2)
    
    # to replace:
    # plt.plot([0, 1], [0, 0], color='grey', marker='o', linestyle='solid', linewidth=5, markersize=2)
    # plt.plot([0, 0.5], [0, math.sqrt(3)/2], color='grey', marker='o', linestyle='solid', linewidth=5, markersize=2)
    # plt.plot([1, 0.5], [0, math.sqrt(3)/2], color='grey', marker='o', linestyle='solid', linewidth=5, markersize=2)
    
    plt.text(-0.04, -0.03, 'g', fontsize = 12)
    plt.text(1.02, -0.03, 'w', fontsize = 12)
    plt.text(0.49, 0.89, 'o', fontsize = 12) # 0.8660254037844386
    
    s_w=S[0,:]; s_g=S[1,:]; s_o=1-s_w-s_g
    # x = s_g+0.5*s_o
    x = s_w+0.5*s_o
    y = math.sqrt(3)/2*s_o
    plt.plot(x, y, color='black', linestyle='dashed', linewidth=1, marker="o", markerfacecolor='black', markersize=8)
    
    plt.axis('off')
    
    # exact solution    
    plot_exact()
    
       
# exact solution        
def plot_exact():
    global a, b, c, mu_w, mu_g, mu_o 
    # a=3.29; b=2.65; c=5.84
    # mu_w=1; mu_g=0.5; mu_o=2    
    t_sol = np.linspace(0, 0.36, 100)
    s_w = (mu_w*t_sol)**(1 / (a-1))
    s_g = (mu_g*t_sol)**(1 / (b-1))
    s_o = 1 - s_w - s_g
    
    # x = s_g+0.5*s_o
    x = s_w+0.5*s_o
    y = math.sqrt(3)/2*s_o
    plt.plot(x, y, color='blue', linestyle='dashed', linewidth=1) # , marker="o", markerfacecolor='black', markersize=8)
    
    return
    
    
def viz_sol_various(x, S):
    # vizualize the solutions
    global fig_num
    global t
    global dir_path
    
    plt.style.use('seaborn-poster')
    
    # plt.figure(figsize = (12, 4))
    plt.figure(figsize = (18, 4))
    
    plt.subplot(141)
    # plt.plot(x, S[0,:])
    plt.plot(x, S[0,:], color='grey', marker='o', markerfacecolor='black', linestyle='solid', linewidth=2, markersize=8)
    
    # variant "all-in-one"
    plt.plot(x, S[0,:], color='blue', marker='o', markerfacecolor='blue', linestyle='solid', linewidth=2, markersize=8)
    plt.plot(x, 1-S[1,:], color='red', marker='o', markerfacecolor='red', linestyle='solid', linewidth=2, markersize=8)
    plt.ylim([0, 1])
    
    plt.xlabel('x')
    plt.ylabel('s_w(x)')
    
    plt.subplot(142)
    # plt.plot(x, S[1,:])
    plt.plot(x, S[1,:], color='grey', marker='o', markerfacecolor='black', linestyle='solid', linewidth=2, markersize=8)  
    
    plt.ylim([0, 1])
    plt.xlabel('x')
    plt.ylabel('s_g(x)')
    
    plt.subplot(143)
    # plt.plot(x, S[1,:])
    # plt.plot(x, 1 - S[0,:] - S[1,:])    
    plt.plot(x, 1 - S[0,:] - S[1,:], color='grey', marker='o', markerfacecolor='black', linestyle='solid', linewidth=2, markersize=8)  
        
    plt.ylim([0, 1])
    plt.xlabel('x')
    plt.ylabel('s_o(x)')
    
    plt.tight_layout()    
    
    # A=[0, 0]; B=[1,0]; C=[0.5, math.sqrt(3)/4]       
    plt.subplot(144)
    plot_triangle(S)
    
    # filename = "figures/test_triangle.png"
    
    filename = dir_path+"/test_triangle_"+ str(fig_num)+"-"+str(round(t,2))+".png"
    fig_num = fig_num + 1
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()
    
    
    
def viz_sol(x, S):
    # vizualize the solutions
    global fig_num
    global t
    global dir_path
    
    plt.style.use('seaborn-poster')
    
    plt.figure(figsize = (12, 4))
    # plt.figure(figsize = (18, 4))
    
    plt.subplot(121)
    # variant "all-in-one"
    plt.plot(x, S[0,:], color='blue', marker='o', markerfacecolor='blue', linestyle='solid', linewidth=2, markersize=8)
    plt.plot(x, 1-S[1,:], color='red', marker='o', markerfacecolor='red', linestyle='solid', linewidth=2, markersize=8)
    plt.ylim([0, 1])
    
    plt.xlabel('x')
    plt.ylabel('s_w(x)')
    
    # A=[0, 0]; B=[1,0]; C=[0.5, math.sqrt(3)/4]       
    plt.subplot(122)
    plot_triangle(S)
    
    # filename = "figures/test_triangle.png"
    filename = dir_path+"/test_triangle"+ str(fig_num)+"-"+str(round(t,2))+".png"
    fig_num = fig_num + 1
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()    
    
####################################################################
#
# Calculate one step
#
####################################################################    
    
def one_step(S):
    global dt, dx
    
    # calculating numerical fluxes
    n_cells = S.shape[1]
    fplus = np.zeros((2, n_cells))
    for k in range(0, n_cells):
        # upwind 
        F = flux_basic(S[:,k])
        # F = flux_improved(S[:,k])
        # F = flux(S[:,k])
        #
        # print('F:[',k ,'] = ', S)
        fplus[0,k] = F[0] 
        fplus[1,k] = F[1]
        
        # central diffences
        
        
        
        
        
        
                   
    # fplus = flux(S) # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all() 
    
    # print('fplus:', fplus)
    #
    # main marching step
    for k in range(1, n_cells):
        S[0,k] = S[0,k] - dt/dx*(fplus[0,k] - fplus[0,k-1])
        S[1,k] = S[1,k] - dt/dx*(fplus[1,k] - fplus[1,k-1])
    
    return S
    
    
####################################################################
#
# Marching
#
####################################################################

# solve_ivp(fun, [0,1], S, method = 'RK45', t_eval = T) 

# def marching(x,T,S):
def marching(x, T, S, t_eval, t_frequency, t_threshold): 
    global t
    # global t_eval, t_frequency, t_threshold
    global dir_path
    
    # the global iteration begins here
    # global SS
    SS = [] # list of 'S'
    S0 = []
    S1 = []
    SS.append(S) # to consider initial data
    S0.append(S[0])
    S1.append(S[1])
    
    # print('S:', S)
    k=0
    
    # evalation times
    # t_eval = np.linspace((t_span[0], t_span[1], 10 )
    # t_eval = np.linspace(T[0], T[-1], 10)
    # t_eval = np.linspace(T[0], T[-1], 50)
    #
    # n_eval = 1000
    # t_eval = np.linspace(T[0], T[-1], 1000)
    file = open(dir_path+"/out.txt", "w+")
    file_0 = open(dir_path+"/out_0.txt", "w+")
    file_1 = open(dir_path+"/out_1.txt", "w+")
    file_x = open(dir_path+"/out_x.txt", "w+")
               
    j=0
    for t in T:
        # output
        if t >= t_eval[j]:
            j=j+1
            C = str(S); file.write(C)
            C = str(S[0]); file_0.write(C)
            C = str(S[1]); file_1.write(C)            
            # print(j, '- S', S)
            print('j=', j,', t=', round(t,2))
            viz_sol(x, S)
            SS.append(S)
            S0.append(S[0])
            S1.append(S[1])            
            # np.savetxt(dir_path+'/a_save.dat', S[0])
            np.savetxt(dir_path+'/a_save_S0.txt', S[0], fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
            # print(S[0], '\n')
            file_x.write(str(S[0])+'\n')
            
        k=k+1
        # print(k, '- S:', S)
                
        # if (t*t_scale - math.floor(t*t_scale)) > 0.34545:  # 0.65454:
        if (t*t_frequency - math.floor(t*t_frequency)) < t_threshold:            
            S[0,0]=1.0
            S[1,0]=0
        else:
            S[0,0]=0
            S[1,0]=1.0
            
        S = one_step(S)
    
    file.close(); 
    file_0.close()
    file_1.close()   
    return SS, S0, S1
    # return x, S
    
####################################################################
#
# Auxilliary
#
####################################################################
#
# parsing float numbers
def get_arg(arg_x, ref_x):
    if arg_x:
        # return_x= float(arg_x[0])
        return_x= arg_x[0]
    else:
        return_x = ref_x
    return return_x
        
####################################################################
#
# Main routine
#
####################################################################
# python corey_plus.py -tmax 0.1 
# python corey_plus.py -mu 1 0.5 2 -abc 3.29 2.65 5.84 -tmax 2.5 -xmax 4 -ncells 100 -dt 0.001 -neval 1000 -tfreq 20 -thresh 0.65454   
# python corey_plus.py -mu 1 0.02957 71.42 -abc 3.29 2.65 5.84 -tmax 2.5 -xmax 4 -ncells 100 -dt 0.001 -neval 1000 -tfreq 20 -thresh 0.65454   
#
# python corey_plus.py -mu 1 0.02957 71.42 -abc 3.29 2.65 5.84 -tmax 0.2 -xmax 4 -ncells 100 -dt 0.001 -neval 50 -tfreq 20 -thresh 0.316


# low frecuency 
# python corey_plus.py -mu 1 0.5 2 -abc 3.29 2.65 5.84 -tmax 1 -xmax 10 -ncells 1000 -dt 0.001 -neval 10 -tfreq 1 -thresh 0.65454   
# python corey_plus.py -mu 1 0.5 2 -abc 3.29 2.65 5.84 -tmax 10 -xmax 10 -ncells 1000 -dt 0.001 -neval 1000 -tfreq 1 -thresh 0.65454
# python corey_plus.py -mu 1 0.5 2 -abc 3.29 2.65 5.84 -tmax 5 -xmax 10 -ncells 1000 -dt 0.001 -neval 100 -tfreq 1 -thresh 0.65454
# python corey_plus.py -mu 1 0.5 2 -abc 3.29 2.65 5.84 -tmax 5 -xmax 10 -ncells 1000 -dt 0.0001 -neval 1000 -tfreq 1 -thresh 0.65454
# python corey_plus.py -mu 1 0.5 2 -abc 3.29 2.65 5.84 -tmax 0.5 -xmax 10 -ncells 100 -dt 0.0001 -neval 100 -tfreq 1 -thresh 0.5
# python corey_plus.py -mu 1 0.5 2 -abc 3.29 2.65 5.84 -tmax 5 -xmax 10 -ncells 1000 -dt 0.001 -neval 1000 -tfreq 1 -thresh 0.5
# python corey_plus.py -mu 1 0.5 2 -abc 3.29 2.65 5.84 -tmax 5 -xmax 5 -ncells 200 -dt 0.001 -neval 1000 -tfreq 4 -thresh 0.5


def main():
    global a, b, c, mu_w, mu_g, mu_o # flux function parameters
    global dt, dx
    global dir_path
    
    global fig_num
    fig_num = 10000
    #
    # python corey-plus.py --mu 1.2 3.4 4.5
    #
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')
    
    # Required positional argument
    # parser.add_argument('pos_arg', type=int, help='A required integer positional argument')
    

    # Optional argument
    # parser.add_argument('--mu', type=str,  nargs='?', help='optional mu values')
    parser.add_argument('-mu', type=float, nargs='*', help='mu specification')
    parser.add_argument('-abc', type=float, nargs='*', help='parameters a, b, c')
    parser.add_argument('-tmax', type=float, nargs='*', help='max t value')
    parser.add_argument('-xmax', type=float, nargs='*', help='max x value')
    parser.add_argument('-ncells', type=int, nargs='*', help='number of cells')
    parser.add_argument('-dt', type=float, nargs='*', help='time step')
    parser.add_argument('-neval', type=int, nargs='*', help='number of stored evaluations')
    parser.add_argument('-tfreq', type=int, nargs='*', help='oscillation frequency')
    parser.add_argument('-thresh', type=float, nargs='*', help='oscillation threshold')
    args = parser.parse_args()
    
    # a, b, c = 3.29, 2.65, 5.84
    # mu_w, mu_g, mu_o = 1, 0.5, 2
    
    #############################################
    # mu
    mu = args.mu    
    if mu:
        if len(mu) < 3:
            print('mu: 3 arguments requireds')
        else:            
            mu_w = float(mu[0])
            mu_g = float(mu[1])
            mu_o = float(mu[2])
    else:
        mu_w, mu_g, mu_o = 1, 0.5, 2
    
    #############################################        
    # abc
    abc = args.abc    
    if abc:
        if len(abc) < 3:
            print('abc: 3 arguments requireds')
        else:
            a = float(abc[0])
            b = float(abc[1])
            c = float(abc[2])
    else:
        a, b, c = 3.29, 2.65, 5.84
        
    #############################################
    #
    # Grid frame
    #        
    #############################################
    t_max  = get_arg(args.tmax, 0.1) # 2.5
    t_span = [0, t_max]
    x_max  = get_arg(args.xmax, 4)
    x_span = [0, x_max]     # Define the grid interval
    n_cells  = get_arg(args.ncells, 100)
    dt  = get_arg(args.dt, 0.01) # 0.001
     
    #############################################
    # marching parameters: t_eval, t_frequency, t_threshold
    #############################################        
    n_eval = get_arg(args.neval, 100) # 1000
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)       
    t_frequency = get_arg(args.tfreq, 2) # 20
    t_threshold = get_arg(args.thresh, 0.65454)
        
    #############################################        
    # dir_path; used for write images and save video
    global path_num
    path_num = 1
    while True:
        dir_path = 'figures/figures-' + str(path_num)
        # print(fig_path_n)
        if not os.path.isdir(dir_path):
            break
        path_num = path_num + 1
    os.mkdir(dir_path)        
    print('directory: ', dir_path) 
        
    # print("Argument values are in file parameters.txt")    
    with open(dir_path+'/'+'parameters-'+str(path_num)+'.txt', "w") as f:
        f.write("a = " + str(a) + "\n" + "b = " + str(b) + "\n" + "c = " + str(c) + "\n" 
                + "mu_w = " + str(mu_w) + "\n" + "mu_g = " + str(mu_g) + "\n" + "mu_o = " + str(mu_o) + "\n" 
                + "t_span = " + str(t_span) + "\n" + "x_span = " + str(x_span) + "\n" + "n_cells = " + str(n_cells) + "\n" + "dt = " + str(dt) + "\n"
                + "n_eval = " + str(n_eval) + "\n" + "t_frequency = " + str(t_frequency) + "\n" + "t_threshold = " + str(t_threshold) + "\n" + "\n \n")        
    
    #############################################
    #
    # run
    #        
    #############################################
    
    # calling_foamflux()
    
    # print('start')
    x, T, S = initialize(x_span, n_cells, t_span, dt)
    SS, S0, S1 = marching(x, T, S, t_eval, t_frequency, t_threshold)
    
    S0 = np.array(S0)
    S1 = np.array(S1)
    
    with open(dir_path+'/'+'S0-'+str(path_num)+'.txt', "w") as f:
        C = str(S0)
        f.write(C)    
        
    
    with open(dir_path+'/'+'S1-'+str(path_num)+'.txt', "w") as f:
        C = str(S1)
        f.write(C)
        
        
    #############################################
    #
    # plot contour
    #        
    #############################################
        
    print('S0.shape:', S0.shape)
    print('len(x):', len(x))
    print('len(t_eval):', len(t_eval))
    
    S0 = np.array(S0)
    S1 = np.array(S1)
    print('S0.shape (array):', S0.shape)
    
    fig, ax = plt.subplots(1, 1)      
    ax.contour(x,t_eval,S0)
    filename = dir_path+"/contour_"+ str(path_num) + "-S0.png"
    fig.savefig(filename, bbox_inches='tight')
    ax.contour(x,t_eval,S1)
    filename = dir_path+"/contour_"+ str(path_num) + "-S1.png"
    fig.savefig(filename, bbox_inches='tight')
    
    fig.show()
    
    #############################################
    #
    # vizualize video
    #        
    #############################################
        
    # viz_sol(x, S)
    save_video(dir_path)
    
    return SS
    

if __name__ == "__main__":
    main()
    
    
    
    
    