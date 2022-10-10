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
    
    #***************************************************************
    #
    #  WARNING:  Cambie de lugar este código, así no hay repetición
    #
    #***************************************************************
    # # caption and projection of outliers
    # if s_w < 0:
    #     s_w = 0
    # if s_g < 0:
    #     s_g = 0
    # if s_w + s_g > 1:
    #     s_norm = (s_w**2 + s_g**2)**0.5
    #     s_w = s_w / s_norm
    #     s_g = s_g / s_norm
    
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
""" The flux is given as a series of jets of the mobilities """

def waterJet(S):
    global a, mu_w
    sw = S[0]
    
    nw = a; muw = mu_w
    
    # Water only have Convex Corey permeability model
    mw   = lambda sw: sw**nw/muw
    dmw  = lambda sw: nw*(sw**(nw - 1))/muw
    
    return mw(sw), dmw(sw)



def gasJet(S):
    global b, mu_g
    global foamM
    sw = S[0]; sg = S[1]
    
    ng = b; mug = mu_g
    
    # Gas only have Convex Corey permeability model
    mg0  = lambda sg: sg**ng/mug
    dmg0 = lambda sg: ng*(sg**(ng - 1))/mug
    
    """ When foam model is activated, some parameters need to be settled """
    if foamM == 'foam':
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
            # Is oil quality changing?
            so = 1 - sw - sg
            if so < floil or so > fmoil:
                dF3 = 0.
            else:
                dF3 = (((fmoil - so)/(fmoil - floil))**(epoil - 1) )/(floil - fmoil)
            # Is water quality changing?
            x  = 2.*epdry*(sw - fmdry)
            if abs(x) >= 1:
                dF2 = 0.
            else:
                dF2 = 1.875*epdry*(x**2 - 1)**2
            fmmobInv = 1/(1. + fmmob*F_2(sw)*F_3(1 - sw - sg))
            dFRMdsw = -fmmob*dF2*F_3(so)*fmmobInv
            dFRMdsg = -fmmob*F_2(sw)*dF3*fmmobInv
            # Some differentials
            dmg = (dmg0(sg) + mg0(sg)*(fmmob*F_2(sw)*dF3)*fmmobInv)*fmmobInv
            return dFRMdsw, dFRMdsg, dmg
        mg  = lambda sw, sg: mg0(sg)/(1. + fmmob*F_2(sw)*F_3(1 - sw - sg))
        dmg = lambda sw, sg: dFRM(sw, sg)[2]
    else:
        mg  = lambda sw, sg: mg0(sg)
        dmg = lambda sw, sg: dmg0(sg)
    
    return mg(sw, sg), dmg(sw, sg)



def oilJet(S):
    global c, mu_o
    global model
    sw = S[0]; sg = S[1]
    
    if model == 'Corey':
        # Corey oil permeability depends upon oil saturation only
        no = c; muo = mu_o
        mo  = lambda sw, sg: (1 - sw - sg)**no / muo
        dmo = lambda sw, sg: no*((1 - sw - sg)**(no - 1)) / muo
        dmodsw = lambda sw, sg: -dmo(sw, sg)
        dmodsg = lambda sw, sg: -dmo(sw, sg)
    elif model == 'Stone':
        # For Stone I model, the oil permeability is different
        nwo = 2.0; ngo = 2.0; muo = mu_o
        mo = lambda sw, sg: (1 - sw - sg)*(1 - sw)**(nwo - 1)*(1 - sg)**(ngo - 1) / muo 
        dmodsw = lambda sw, sg: -( (1 - sw)**(nwo - 1) + (nwo - 1)*(1 - sw - sg)*((1 - sw)**(nwo - 2)))*(1 - sg)**(ngo - 1) / muo
        dmodsg = lambda sw, sg: -( (1 - sg)**(ngo - 1) + (ngo - 1)*(1 - sw - sg)*((1 - sg)**(ngo - 2)))*(1 - sw)**(nwo - 1) / muo
    
    return mo(sw, sg), dmodsw(sw, sg), dmodsg(sw, sg)
    
    

def jet(S):
    global a, b, c, mu_w, mu_g, mu_o
    sw = S[0]; sg = S[1]

    #***************************************************************
    #
    #  WARNING:  Cambie de lugar este código, así no hay repetición
    #
    #***************************************************************
    # # caption and projection of outliers
    # if sw < 0:
    #     sw = 0
    # if sg < 0:
    #     sg = 0
    # if sw + sg > 1:
    #     s_norm = (sw**2 + sg**2)**0.5
    #     sw = sw / s_norm
    #     sg = sg / s_norm    
    # S = [sw, sg]

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
    mw, dmw = waterJet(S)
    mg, dmg = gasJet(S)
    mo, dmodsw, dmodsg = oilJet(S)
    
    # Now we do all needed computations
    mTotal = mw + mg + mo
    mT_inv = 1/mTotal
    dmTdsw = dmw + dmodsw
    dmTdsg = dmg + dmodsg
    
    """ The fluxes are calculated first """
    fw = mw*mT_inv
    fg = mg*mT_inv
    
    """ Then the jacobian of the flux """
    fww = (dmw*mTotal - dmTdsw*mw)*mT_inv**2
    fwg = -dmTdsg*mw*mT_inv**2
    fgw = -dmTdsw*mg*mT_inv**2
    fgg = (dmg*mTotal - dmTdsg*mg)*mT_inv**2
    
    # The jacobian entrances
    detJ = fww*fgg - fwg*fgw    # Thus the determinat of th Jacobian
    traJ = fww + fgg            # and  the trance of the same matrix
    
    return fw, fg, fww, fwg, fgw, fgg, detJ, traJ



def flux_improved(S):
    JET = jet(S)
    
    F = np.array([JET[0], JET[1]])
    return F




# The Jacobian for flux function for Corey/Stone I model
def Jacobian(S):
    JET = jet(S)
    
    fww = JET[2]
    fwg = JET[3]
    fgw = JET[4]
    fgg = JET[5]
    
    J = np.array([[fww, fwg], [fgw, fgg]])
    return J


# def Jacobian_test(S, h):
#     Jexact = Jacobian(S)
#     FN = flux(S + h*np.array([0, 1]))
#     FS = flux(S - h*np.array([0, 1]))
    
#     FE = flux(S + h*np.array([1, 0]))
#     FW = flux(S - h*np.array([1, 0]))
    
#     hinv2 = 1./(2.*h)
#     Jnum = np.array([[(FE[0] - FW[0])*hinv2, (FN[0] - FS[0])*hinv2], [(FE[1] - FW[1])*hinv2, (FN[1] - FS[1])*hinv2]])
    
#     return Jexact - Jnum, Jexact, Jnum
#########################################################################################################


#%%
####################################################################
#
# Analytical solution
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


#%%
def bisection(f, n):
    # The initial interval is [0, 1], for a tolerance 10**(-n), 
    # the next iterations are fine.
    a = 0.; b = 1.
    fa = f(a); fb = f(b)
    for k in range( int( np.ceil(0.69315*n) ) ):
        c = (a + b)/2
        fc = f(c)
        if fc*fb > 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
    return (a + b)/2
                
#%%
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

""" This will be the engine to solve the WAG injection in the semi-
    anlytical form for a given injection L into a reservoir with
    rihgt state R as pure oil, R = (0, 0).
"""
def lambda1(S):
    fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet(S)
    lambda_s = (traJ - np.sqrt(traJ**2 - 4*detJ))/2
    return lambda_s

def lambda2(S):
    fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet(S)
    lambda_f = (traJ + np.sqrt(traJ**2 - 4*detJ))/2
    return lambda_f


def extW(sw):
    S = [sw, 0.]
    mo, dmow, dmog = oilJet(S)
    mw, dmw = waterJet(S)
    
    return sw*(mo*dmw - dmow*mw) - mw*(mo + mw)


def extG(sg):
    S = [0., sg]
    mo, dmow, dmog = oilJet(S)
    mg, dmg = gasJet(S)

    return sg*(mo*dmg - dmog*mg) - mg*(mo + mg)



def Bstar(S):
    mg, dmg = gasJet(S)
    mw, dmw = waterJet(S)
    
    fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet(S)
    lambda_s = (traJ - np.sqrt(traJ**2 - 4*detJ))/2
    
    return fw - S[0]*lambda_s, S[1]*mw - S[0]*mg



def nullOil(t, S):
    return 1 - S[0] - S[1]

def charGasShock(t, S):
    fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet(S)
    lambda_s = (traJ - np.sqrt(traJ**2 - 4*detJ))/2
    return fg - S[1]*lambda_s

def charWaterShock(t, S):
    fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet(S)
    lambda_s = (traJ - np.sqrt(traJ**2 - 4*detJ))/2
    return fw - S[0]*lambda_s


def bwdRar(t, S):
    fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet(S)
    lambda_s = (traJ - np.sqrt(traJ**2 - 4*detJ))/2
    norm = -np.sqrt((fww - lambda_s)**2 + fwg**2)
    return [fwg/norm, (lambda_s - fww)/norm]

def fwdRar(t, S):
    fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet(S)
    lambda_s = (traJ - np.sqrt(traJ**2 - 4*detJ))/2
    
    norm = np.sqrt((fww - lambda_s)**2 + fwg**2)
    return [fwg/norm, (lambda_s - fww)/norm]



def prove_Bstar(Bs):
    fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet(Bs)
    Ow, Og, Oww, Owg, Ogw, Ogg, detO, traO = jet([0., 0.])
    
    lam1  = (traJ - np.sqrt(traJ**2 - 4*detJ))/2
    sigmaW = (fw - Ow)/Bs[0]
    sigmaG = (fg - Og)/Bs[1]
        
    print(lam1 - sigmaW, lam1 - sigmaG)
    
    return



def prove_BW_point(Star):
    fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet(Star)
    Ow, Og, Oww, Owg, Ogw, Ogg, detO, traO = jet([0., 0.])
    
    lam2  = (traJ + np.sqrt(traJ**2 - 4*detJ))/2
    sigmaW = (fw - Ow)   # /Star[0]
    sigmaG = (fg - Og)   # /Star[1]
    
    print(lam2*Star[0] - sigmaW, lam2*Star[1] - sigmaG)
    
    return




def WAG_solution(L, R = np.array([0, 0]).reshape((2, 1))):
    global foamM, fmoil
    global mu_w, mug_g, a, b
    global B, Ws, lam_Ws, Gs, lam_Gs
    global rar1I, rar2I, lam1I, lam2I, sigma2I, satR1, satR2
    global breakthrough
    
    muw = mu_w; mug = mu_g
    
    if L[0] > B[0]:
        breakthrough = 'water_side'
        charGasShock.terminal = True
        charGasShock.direction = -1
        Irar1 = solve_ivp(fwdRar, [0., 0.5], L, vectorized = True, events = charGasShock, t_eval = np.linspace(0., 0.5, 50))
        rar1I = Irar1.y
        # The characteristic speed is given by the slow-family
        fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet(rar1I)
        lam1I = (traJ - np.sqrt(traJ**2 - 4*detJ))/2
        fwE = fw[-1]; Ew = rar1I[0, -1]
        def middleWO(s):
            fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet([s, 0.])
            return fw - fwE - (Ew - s)*lam1I[-1]
        m = float(fsolve(middleWO, 0.5))
        if Ws < m:
            rar2I = np.empty([51, 2])
            h = (Ws - m)/50
            for k in range(51):
                rar2I[k, :] = [k*h + m, 0.]
            sigma2I = lam_Ws
            rar2I = rar2I.T
            # The characteristic speed is given by the fast-family
            fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet(rar2I)
            lam2I = (traJ + np.sqrt(traJ**2 + 4*detJ))/2
        else:
            rar2I = [m, 0.]
            mw, dmw            = waterJet(rar2I)
            mo, dmodsw, dmodsg = oilJet(rar2I)
            rar2I = np.array([[m, m], [0., 0.]])
            sigma2I = mw/((mw+ mo)*m)
            lam2I = np.array([sigma2I, sigma2I])
    else:
        breakthrough = 'gas_side'
        charWaterShock.terminal = True
        charWaterShock.direction = -1
        Irar1 = solve_ivp(fwdRar, [0., 0.5], L, vectorized = True, events = charWaterShock, t_eval = np.linspace(0., 0.5, 50))
        rar1I = Irar1.y
        # The characteristic speed is given by the slow-family
        fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet(rar1I)
        lam1I = (traJ - np.sqrt(traJ**2 - 4*detJ))/2
        fgE = fg[-1]; Eg = rar1I[1, -1]
        def middleGO(s):
            fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet([0., s])
            return fg - fgE - (Eg - s)*lam1I[-1]
        m = float(fsolve(middleGO, 0.5))
        if Gs < m:
            rar2I = np.empty([51, 2])
            h = (Gs - m)/50
            for k in range(51):
                rar2I[k, :] = [0., k*h + m]
            sigma2I = lam_Gs
            rar2I = rar2I.T
            # The characteristic speed is given by the fast-family
            fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet(rar2I)
            lam2I = (traJ + np.sqrt(traJ**2 + 4*detJ))/2
        else:
            rar2I = [0., m]
            mg, dmg            = gasJet(rar2I)
            mo, dmodsw, dmodsg = oilJet(rar2I)
            rar2I = np.array([[0., 0.], [m, m]])
            sigma2I = mg/((mg + mo)*m)
            lam2I = np.array([sigma2I, sigma2I])
    satR1 = triangle(rar1I[0, :], rar1I[1, :])
    satR2 = triangle(rar2I[0, :], rar2I[1, :])
    return



def WAG_optimal():
    global foamM, fmoil
    global mu_w, mug_g, a, b
    global lam_rar1, rar1, Bs, lam_Bs
    global B, Ws, lam_Ws, Gs, lam_Gs
    global breakthrough
    
    powW = a - 1; epog = 1/(b - 1); mugw = mu_g/mu_w
    
    """ Some notation of this code is explained in Castañeda et al.
        (2016), and may be found there.
    """
    
    # Behte-Wendroff points along binary edges GO, WO, the "s" stads for star
    Ws = bisection(extW, 100)
    TWs = triangle([Ws], [0])
    # The characteristic speed is given by the fast-family
    fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet([Ws, 0.])
    lam_Ws  = (traJ + np.sqrt(traJ**2 - 4*detJ))/2 
    
    Gs = bisection(extG, 100)
    TGs = triangle([0], [Gs])
    # The characteristic speed is given by the fast-family
    fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet([0., Gs])
    lam_Gs  = (traJ + np.sqrt(traJ**2 - 4*detJ))/2
    
    # The internal Hugonot locus from pure oil gives B_star
    Bs = fsolve(Bstar, [0.7*Ws, 0.7*Gs])
    TBs = triangle([Bs[0]], [Bs[1]])
    # The characteristic speed is given by the slow-family
    fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet(Bs)
    lam_Bs  = (traJ - np.sqrt(traJ**2 - 4*detJ))/2
    
    ################################################################
    #
    #   HERE we can test all Bethe-Wendroff points, G*, B* and W*
    #
    ################################################################
    #
    # print(lam_Gs, lam_Bs, lam_Ws)
    # print("Water- and Gas-star:", Ws, Gs)
    # prove_BW_point([Ws, 0.])
    # prove_BW_point([0., Gs])
    # print("Optimal Bethe-Wendroff", Bs)
    # prove_Bstar(Bs)
    #
    ################################################################
    
    if foamM == 'foam' and Bs[0] + Bs[1] > 1. - fmoil:
        print('WARNING: the optimal solution is committed to the current parameters.')
    
    swH = []; sgH = []
    for s in np.linspace(0, Bs[0], 30):
        swH.append(s); sgH.append( (mugw*s**powW)**epog )
    intH = triangle(swH, sgH)    
    
    # The backward rarefaction from Bs
    nullOil.terminal = True
    nullOil.direction = -1
    rar1 = solve_ivp(bwdRar, [0., 1.], Bs, vectorized = True, events = nullOil, t_eval = np.linspace(0., 1., 100))
    B = [rar1.y[0, -1], rar1.y[1, -1]]
    
    # The characteristic speed is given by the slow-family
    fw, fg, fww, fwg, fgw, fgg, detJ, traJ = jet(rar1.y)
    lam_rar1 = (traJ - np.sqrt(traJ**2 - 4*detJ))/2
    satR = triangle(rar1.y[0, :], rar1.y[1, :])
    
    plt.plot([TWs[0], TBs[0], TGs[0]], [TWs[1], TBs[1], TGs[1]], color = 'gray', linestyle = 'dashed', linewidth = 0.5)

    return intH, satR, Bs, B

#%%


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
    
    plt.text(-0.060, -0.03, 'G', fontsize = 12)
    plt.text( 1.025, -0.03, 'W', fontsize = 12)
    plt.text( 0.480,  0.89, 'O', fontsize = 12) # 0.8660254037844386
    
    s_w = S[0,:]; s_g = S[1,:]; s_o = 1 - s_w - s_g
    # x = s_g+0.5*s_o
    x = s_w + 0.5*s_o
    y = math.sqrt(3)/2*s_o
    plt.plot(x, y, color='black', linestyle='dashed', linewidth=1, marker="o", markerfacecolor='black', markersize=8)
    
    plt.axis('off')
    
    # exact solution    
    plot_exact()
    
    return
    
       
# exact solution        
def plot_exact():
    global satR1, rar2I, lam1I, sigma2I
    intH, satR, Bs, B = WAG_optimal()
    
    # Backward internal Hugoniot from pure Oil
    plt.plot(intH[0, :], intH[1, :], color='blue', linestyle='dashed', linewidth = 1)
    # Backward slow-family rarefaction from Bethe-Wendroff point
    plt.plot(satR[0, :], satR[1, :], color='blue', linewidth = 1)
    
    # Special solution
    plt.plot(satR1[0, :], satR1[1, :], color='blue', linewidth = 1)
    plt.plot([satR1[0, -1], satR2[0, 0]], [satR1[1, -1], satR2[1, 0]], color='blue', linestyle='dashed', linewidth = 1)
    plt.plot(satR2[0, :], satR2[1, :], color='red', linewidth = 1)
    plt.plot([satR2[0, -1], 0.5], [satR2[1, -1], 0.8667], color='red', linestyle='dashed', linewidth = 1)

    return


def viz_exactSol(x, t):
    global rar1I, rar2I, lam1I, sigma2I
    global breakthrough
    
    # Rarefaction slow-family
    plt.plot(t*lam1I,      rar1I[0], color = 'gray', linestyle = 'dashed')
    plt.plot(t*lam1I, 1. - rar1I[1], color = 'gray', linestyle = 'dashed')

    # Lax-1 Shock wave characteristic to the boundary with middle state
    if lam1I[-1] > lam2I[0]:
        print("WARNING (", breakthrough,"): incompatible characteristic values at the midpoint.")
        print('         lambda_1 =', lam1I[-1], " lambda_2 =", lam2I[0])
    plt.plot([t*lam1I[-1], t*lam1I[-1], t*lam2I[0]],      [rar1I[0, -1],      rar2I[0, 0],      rar2I[0, 0]], color = 'gray', linestyle = 'dashed')
    plt.plot([t*lam1I[-1], t*lam1I[-1], t*lam2I[0]], [1. - rar1I[1, -1], 1. - rar2I[1, 0], 1. - rar2I[1, 0]], color = 'gray', linestyle = 'dashed')

    # Rarefaction fast-family together
    plt.plot(t*lam2I,      rar2I[0, :], color = 'gray', linestyle = 'dashed')
    plt.plot(t*lam2I, 1. - rar2I[1, :], color = 'gray', linestyle = 'dashed')

    # Lax-2 Shock wave characteristic to the boundary
    plt.plot([t*sigma2I, t*sigma2I, x[-1]],      [rar2I[0, -1], 0., 0.], color = 'gray', linestyle = 'dashed')
    plt.plot([t*sigma2I, t*sigma2I, x[-1]], [1. - rar2I[1, -1], 1., 1.], color = 'gray', linestyle = 'dashed')

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
    
    plt.xlabel('$x$')
    plt.ylabel('$s_w(x)$')
    
    plt.subplot(142)
    # plt.plot(x, S[1,:])
    plt.plot(x, S[1,:], color='grey', marker='o', markerfacecolor='black', linestyle='solid', linewidth=2, markersize=8)  
    
    plt.ylim([0, 1])
    plt.xlabel('$x$')
    plt.ylabel('$s_g(x)$')
    
    plt.subplot(143)
    # plt.plot(x, S[1,:])
    # plt.plot(x, 1 - S[0,:] - S[1,:])    
    plt.plot(x, 1 - S[0,:] - S[1,:], color='grey', marker='o', markerfacecolor='black', linestyle='solid', linewidth=2, markersize=8)  
        
    plt.ylim([0, 1])
    plt.xlabel('$x$')
    plt.ylabel('$s_o(x)$')
    
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
    global lam_rar1, rar1, Bs, lam_Bs
    
    plt.style.use('seaborn-poster')
    
    plt.figure(figsize = (12, 4))
    # plt.figure(figsize = (18, 4))
    
    # A=[0, 0]; B=[1,0]; C=[0.5, math.sqrt(3)/4]       
    plt.subplot(122)
    plot_triangle(S)
    
    plt.subplot(121)
    # variant "all-in-one"
    plt.plot(x, S[0, :],     color = 'blue', marker = 'o', markerfacecolor = 'blue', linestyle = 'solid', linewidth = 2, markersize = 8)
    plt.plot(x, 1 - S[1, :], color =  'red', marker = 'o', markerfacecolor =  'red', linestyle = 'solid', linewidth = 2, markersize = 8)
    plt.ylim([0, 1])
    
    viz_exactSol(x, t)
#    # Plot the exact solution along the same time t
#    # Rarefaction
#    plt.plot(t*lam_rar1,     rar1.y[0], color = 'gray', linestyle = 'dashed')
#    plt.plot(t*lam_rar1, 1 - rar1.y[1], color = 'gray', linestyle = 'dashed')
#    # Shock wave
#    plt.plot([t*lam_Bs, t*lam_Bs, x[-1]],      [Bs[0], 0., 0.], color = 'gray', linestyle = 'dashed')
#    plt.plot([t*lam_Bs, t*lam_Bs, x[-1]], [1. - Bs[1], 1., 1.], color = 'gray', linestyle = 'dashed')
    
    plt.xlabel('$x$')
    plt.ylabel('$s_w(x)$')
    
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

        #***************************************************************
        #
        #  WARNING:  Cambie de lugar este código, así no hay repetición
        #
        #***************************************************************        
        # caption and projection of outliers
        sw = S[0, k]; sg = S[1, k]
        if sw < 0:
            sw = 0
        if sg < 0:
            sg = 0
        if sw + sg > 1:
            s_norm = (sw**2 + sg**2)**0.5
            sw = sw / s_norm
            sg = sg / s_norm
        S_in_F = [sw, sg]
        
        F = flux_improved(S_in_F)
        # F = flux_basic(S[:,k])
        # F = flux_improved(S[:, k])
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
    global rar1I, rar2I, lam1I, sigma2I
    global leftS
    
    # the global iteration begins here
    # global SS
    SS = [] # list of 'S'
    S0 = []
    S1 = []
    SS.append(S) # to consider initial data
    S0.append(S[0])
    S1.append(S[1])
    
    # print('S:', S)
    k = 0
    
    # evalation times
    # t_eval = np.linspace((t_span[0], t_span[1], 10 )
    # t_eval = np.linspace(T[0], T[-1], 10)
    # t_eval = np.linspace(T[0], T[-1], 50)
    #
    # n_eval = 1000
    # t_eval = np.linspace(T[0], T[-1], 1000)
    file   = open(dir_path+"/out.txt", "w+")
    file_0 = open(dir_path+"/out_0.txt", "w+")
    file_1 = open(dir_path+"/out_1.txt", "w+")
    file_x = open(dir_path+"/out_x.txt", "w+")
    
    intH, satR, Bs, B = WAG_optimal()
    WAG_solution(leftS)
                   
    j = 0
    for t in T:
        # output
        if t >= t_eval[j]:
            j = j + 1
            C = str(S); file.write(C)
            C = str(S[0]); file_0.write(C)
            C = str(S[1]); file_1.write(C)            
            # print(j, '- S', S)
            print('j=', j,', t=', round(t,2))
            viz_sol(x, S)
#            viz_exactSol(x, t)
            SS.append(S)
            S0.append(S[0])
            S1.append(S[1])            
            # np.savetxt(dir_path+'/a_save.dat', S[0])
            np.savetxt(dir_path+'/a_save_S0.txt', S[0], fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
            # print(S[0], '\n')
            file_x.write(str(S[0])+'\n')
            
        k = k + 1
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
    global model, foamM, fmoil
    global leftS
    
    model = 'Corey'
    foamM = 'no-foam'
    
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
    parser.add_argument('-coInject', type=float, nargs='*', help='coinjected saturations')
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
        # a, b, c = 3.29, 2.65, 5.84
        a, b, c = 2, 2, 2
        
    #############################################
    #
    # Grid frame
    #        
    #############################################
    t_max  = get_arg(args.tmax, 5) # 2.5
    t_span = [0, t_max]
    x_max  = get_arg(args.xmax, 10)
    x_span = [0, x_max]     # Define the grid interval
    n_cells  = get_arg(args.ncells, 1000)
    dt  = get_arg(args.dt, 0.001) # 0.001
     
    #############################################
    # marching parameters: t_eval, t_frequency, t_threshold
    #############################################        
    n_eval = get_arg(args.neval, 15) # 1000
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)       
    t_frequency = get_arg(args.tfreq, 2) # 20
    t_threshold = get_arg(args.thresh, 0.65454)
    
    coInject = args.coInject
    if coInject:
        leftS = np.array([coInject[0], coInject[1]])
    else:
        leftS = np.array([t_threshold, 1. - t_threshold])
    
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
#    ax.contour(x,t_eval,S0)
    filename = dir_path+"/contour_"+ str(path_num) + "-S0.png"
    fig.savefig(filename, bbox_inches='tight')
#    ax.contour(x,t_eval,S1)
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
    
    
    
    
    
