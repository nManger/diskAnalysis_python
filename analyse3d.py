from pyPLUTO import nlast_info
import pyPLUTO.pload as pp
import numpy as np
import json
import tools3d as too
import math as m



def alphaRphiAverage(**kwargs):
    ''' calculate \alpha_r,phi disk viscosity parameter 
    
    Parameters
    ----------  
    tbeg    : begin time value in file numbers (default:0)
    tend    : end time value in file numbers (default: nlast)
    **kwargs:  kwargs for setCompRange() in tools3d
    
    Returns 
    -------
    Dictionary with alpha averages:
    Keywords: merAv -- 2D alpha values map, averaged over phi direction
              RmerAv -- 1D alpha values in meridional directions, average over r and phi direction
              Timeser -- Alpha values timeseries, averages over all spatial directions
              TimeserMov -- Alpha values moving average timeseries, averages over all spatial directions
    '''

    #fetch time info and set analysis times
    timeInfo = nlast_info()
    
    tbeg = kwargs.get('tbeg',0)
    tend = kwargs.get('tend',timeInfo['nlast'])
    dt = tend-tbeg+1
    
    rhoav = kwargs.get('rhoav',False)

    #fetch grid info and set analysis domain
    data = pp.pload(0)
    
    data = too.setCompRange(data,**kwargs)
    
    #set data structure & define arrays
    nx1 = getattr(data,'n1')
    nx2 = getattr(data,'n2')
    nx3 = getattr(data,'n3')
    nx1m1 = 1.0/(1.0*nx1)
    nx2m1 = 1.0/(1.0*nx2)
    nx3m1 = 1.0/(1.0*nx3)
    dx1 = data.dx1
    dx2 = data.dx2
    dx3 = data.dx3[None,None,:]
    #dx1 = data.x1*data.x1*data.dx1
    deltax1= np.sum(data.dx1)
#     deltax2=np.sum(data.dx2)
    deltax3=np.sum(data.dx3)
    #deltax1 = np.sum(dx1)

    rhomean = np.zeros((nx1,nx2))
    rhota = np.zeros((nx1,nx2))
    rho1d = np.zeros(nx2)
    rho1dta = np.zeros(nx2)
    vx1vx3rhomean = np.zeros((nx1,nx2))
    vx1vx3rhota = np.zeros((nx1,nx2))
    vx1rhomean = np.zeros((nx1,nx2))
    vx1rhota = np.zeros((nx1,nx2))
    vx3mean = np.zeros((nx1,nx2))
    vx3ta = np.zeros((nx1,nx2))
    csmean = np.zeros((nx1,nx2))
    csta = np.zeros((nx1,nx2))
    alpha2d = np.zeros((nx1,nx2))
    alpha2dta =np.zeros((nx1,nx2))
    alpha1dth = np.zeros(nx2)
    alpha1dmov = np.zeros(nx2)
    alphaTs = np.zeros(dt)
    alphaTsMov = np.zeros(dt)
    
    for tt in range(0,dt):
        data = pp.pload(tt+tbeg)
        data = too.setCompRange(data,**kwargs)
        ntm1 = 1.0/(1.0+tt)
        
        # calculating phi,time averaged alpha(r,phi)
        rhomean       = np.sum(data.rho, axis=2)*nx3m1
        csmean        = np.sum(np.sqrt(data.prs/data.rho),axis=2)*nx3m1          
        vx1vx3rhomean = np.sum(data.vx1*data.vx3*data.rho*dx3,axis=2)/deltax3
        vx1rhomean    = np.sum(data.vx1*data.rho*dx3,axis=2)/deltax3
        if rhoav:
            vx3mean = np.sum(data.vx3*data.rho, axis=2)*nx3m1/rhomean
        else:
            vx3mean = np.sum(data.vx3, axis=2)*nx3m1

        rhota += rhomean
        vx1vx3rhota += vx1vx3rhomean
        vx1rhota += vx1rhomean
        vx3ta += vx3mean
        csta += csmean
        
        alpha2d = (vx1vx3rhomean - vx1rhomean*vx3mean)/(csmean*csmean*rhomean)
        alpha2dta = (vx1vx3rhota*ntm1 - vx1rhota*ntm1*vx3ta*ntm1)/(csta*ntm1*csta*ntm1*rhota*ntm1)
        
        # calculating density averaged radial average of alpha(r,theta)
#         rho1d     = np.sum(rhomean[:,j]*dx1)/deltax1 
#         rho1dta   = np.sum(rhota[:,j]*dx1)*dtm1/deltax1
#         alpha1dth  = np.sum(alpha2d*rhomean*dx1[:,None], axis=0)/np.sum(rhomean*dx1[:,None], axis=0)
#         alpha1dmov = np.sum(alpha2dta*rhota*dx1[:,None], axis=0)/np.sum(rhota*dx1[:,None], axis=0)
        
        # calculating density averaged r,theta average of alpha
        alphaTs[tt] = np.sum(alpha2d*rhomean*(data.x1*dx1)[:,None]*dx2[None,:])/np.sum(rhomean*(data.x1*dx1)[:,None]*dx2[None,:])
        alphaTsMov[tt] = np.sum(alpha2dta*rhota*ntm1*(data.x1*dx1)[:,None]*dx2[None,:])/np.sum(rhota*ntm1*(data.x1*dx1)[:,None]*dx2[None,:])
                
    ntm1 = 1.0/(1.0+dt)
    # radial average of <alpha>_phi,t
    alpha2d = (vx1vx3rhota*ntm1 - vx1rhota*ntm1*vx3ta*ntm1)/(csta*ntm1*csta*ntm1*rhota*ntm1)
    alpha1dth  = np.sum(alpha2d*rhota*ntm1*dx1[:,None], axis=0)/np.sum(rhota*ntm1*dx1[:,None], axis=0)
    
    alphaDict = {}
    
    alphaDict["merAv"] = alpha2d
    alphaDict["RmerAv"]= alpha1dth
    alphaDict["Timeser"]= alphaTs
    alphaDict["TimeserMov"]= alphaTsMov
    
    return alphaDict

def alphaZphiAverage(gamma,q,**kwargs):
    ''' calculate \alpha_z,phi disk viscosity proxy for vsi profile with constant temp. slope q.
    UNDER CONSTRUCTION!!!!! 
    
    Parameters
    ----------
    gamma   : adiabatic constant
    q       : Temperature gradient slope 
    tbeg    : begin time value in file numbers (default:0)
    tend    : end time value in file numbers (default: nlast)
    **kwargs:  kwargs for setCompRange() in tools3d
    
    '''

    #fetch time info and set analysis times
    timeInfo = nlast_info()
    
    tbeg = kwargs.get('tbeg',0)
    tend = kwargs.get('tend',timeInfo['nlast'])
    dt = tend-tbeg+1
    

    #fetch grid info and set analysis domain
    data = pp.pload(0)
    
    data = too.setCompRange(data,**kwargs)
    
    #set data structure & define arrays
    nx1 = getattr(data,'n1')
    nx2 = getattr(data,'n2')
    nx3 = getattr(data,'n3')
    nx1m1 = 1.0/(1.0*nx1)
    nx2m1 = 1.0/(1.0*nx2)
    nx3m1 = 1.0/(1.0*nx3)
    dx1 = data.dx1
    dx3 = data.dx3[None,None,:]
    deltax1 = np.sum(data.dx1)
    deltax3 = np.sum(data.dx3)

    rhomean = np.zeros((nx1,nx2))
    rhota = np.zeros((nx1,nx2))
    rho1d = np.zeros(nx2)
    rho1dta = np.zeros(nx2)
    vx2vx3rhomean = np.zeros((nx1,nx2))
    vx2vx3rhota = np.zeros((nx1,nx2))
    vx2rhomean = np.zeros((nx1,nx2))
    vx2rhota = np.zeros((nx1,nx2))
    vx3mean = np.zeros((nx1,nx2))
    vx3ta = np.zeros((nx1,nx2))
    csmean = np.zeros((nx1,nx2))
    csta = np.zeros((nx1,nx2))
    alpha2d = np.zeros((nx1,nx2))
    alpha2dta =np.zeros((nx1,nx2))
    alpha1dth = np.zeros(nx2)
    alpha1dmov = np.zeros(nx2)
    alphaTs = np.zeros(dt)
    alphaTsMov = np.zeros(dt)
    
    for tt in range(0,dt):
        data = pp.pload(tt+tbeg)
        data = too.setCompRange(data,**kwargs)
        ntm1 = 1.0/(1.0+tt)
        
        # calculating phi,time averaged alpha(r,phi)
        rhomean    = np.sum(data.rho,axis=2)*nx3m1
        vx2vx3rhomean = np.sum(data.vx2*data.vx3*data.rho*dx3,axis=2)/deltax3
        vx2rhomean = np.sum(data.vx2*data.rho*dx3,axis=2)/deltax3
        vx3mean = np.sum(data.vx3*dx3,axis=2)/deltax3
        csmean = np.sum(np.sqrt(gamma*data.prs/data.rho)*dx3,axis=2)/deltax3 
                
        rhota += rhomean
        vx2vx3rhota += vx2vx3rhomean
        vx2rhota += vx2rhomean
        vx3ta += vx3mean
        csta += csmean
        
        # z/R = cos(th)/sin(th)=0 at z=0. Use max(denominator,1e-20) to compensate
        alpha2d = (vx2vx3rhomean - vx2rhomean*vx3mean)/np.max(csmean*csmean*rhomean*q/2.*(np.cos(data.x2)/np.sin(data.x2))[None,:],1.0e-20)
        alpha2dta = (vx2vx3rhota*ntm1 - vx2rhota*ntm1*vx3ta*ntm1)/np.max(csta*ntm1*csta*ntm1*rhota*ntm1*q/2.*(np.cos(data.x2)/np.sin(data.x2))[None,:],1.0e-20)
        
        # calculating density averaged radial average of alpha(r,theta)
#         rho1d       = np.sum(rhomean*dx1[:,None],axis=0)/deltax1 
#         rho1dta     = np.sum(rhota*dx1[:,None],axis=0)*dtm1/deltax1
#         alpha1dth   = np.sum(alpha2d*dx1[:,None],axis=0)/deltax1
#         alpha1dmov  = np.sum(alpha2dta*dx1[:,None],axis=0)/deltax1
        
        # calculating density averaged r,theta average of alpha
        alphaTs[tt] = np.sum(alpha2d*rhomean*(data.x1*dx1)[:,None]*dx2[None,:])/np.sum(rhomean*(data.x1*dx1)[:,None]*dx2[None,:])
        alphaTsMov[tt] = np.sum(alpha2dta*rhota/ntm1*(data.x1*dx1)[:,None]*dx2[None,:])/np.sum(rhota/ntm1*(data.x1*dx1)[:,None]*dx2[None,:])
                
    ntm1 = 1.0/(1.0+dt)
    alpha2d = (vx2vx3rhota*ntm1 - vx2rhota*ntm1*vx3ta*ntm1)/np.max(csta*ntm1*csta*ntm1*rhota*ntm1*q/2.*(np.cos(data.x2)/np.sin(data.x2))[None,:],1.0e-20)
    # radial average of <alpha>_phi,t
    alpha1dth  = np.sum(alpha2d*rhota/ntm1*dx1[:,None], axis=0)/np.sum(rhota/ntm1*dx1[:,None], axis=0)

    alphaDict = {}
    
    alphaDict["merAv"] = alpha2d
    alphaDict["RmerAv"]= alpha1dth
    alphaDict["Timeser"]= alphaTs
    alphaDict["TimeserMov"]= alphaTsMov
    
    return alphaDict


def kappaSq(data):
    ''' calculate the epicyclic frequency of a disk 
    
    Parameters
    ----------
    data: a pyPLUTO data object

    return
    ------
    kappasq: 3d array of epicyclic frequencies
    omega: 3d array of angular frequencies
    
    '''
    
    omega = np.zeros((data.n1,data.n2,data.n3))
    omegadr = np.zeros((data.n1,data.n2,data.n3))
    kappasq = np.zeros((data.n1,data.n2,data.n3))

    for i in data.irange:
        for j in data.jrange:
            for k in data.krange:
                omega[i,j,k] = data.vx3[i,j,k]/data.x1[i]
   # print 'Done omega'
                
    for i in range(1,data.n1-1):
        for j in data.jrange:
            for k in data.krange:
                #print i,j,k
                #print omega[i,j,k]
                omegadr[i,j,k] = (omega[i+1,j,k] - omega[i-1,j,k])/(data.x1[i+1]-data.x1[i-1])
                kappasq[i,j,k] = 4.0*omega[i,j,k]*omega[i,j,k] + 2*data.x1[i]*omega[i,j,k]*omegadr[i,j,k]

    omegadr[0,:,:] = omegadr[1,:,:]
    kappasq[0,:,:] = kappasq[1,:,:]

    omegadr[data.n1-1,:,:] = omegadr[data.n1-2,:,:]
    kappasq[data.n1-1,:,:] = kappasq[data.n1-2,:,:]
    
    return kappasq,omega


def rossbyL(data,gamma=1.0):
    ''' Calculate the function L(R) indicating rossby instability at its extrema
    
    Parameters:
    ----------
    data: a pyPLUTO data object
    gamma: the adiabatic index of the model

    '''
    
    ksq,omeg = kappaSq(data)
    
    rhor = np.average(data.rho,axis=(1,2))
    prsr = np.average(data.prs,axis=(1,2))
    ksqr = np.average(ksq,axis=(1,2))
    omegr = np.average(omeg,axis=(1,2))
           
    L = rhor*omegr/ksqr * np.power(prsr/np.power(rhor,gamma),2.0/gamma)
    
    return L

def rossbyLTser(**kwargs):
    ''' Calculate the function L(R,T) indicating rossby instability at its extrema
    
    Parameters:
    ----------
    gamma: the adiabatic index of the model
    tbeg: begin timestep, default is 0
    tend: end timestep, default is nlast
    kwargs: all other kwargs go to setCompRange in tools3d

    return:
    ------
    L: function L as function of radius and time coordinates
    '''
    
    gamma = kwargs.get('gamma',1.0)

    #fetch time info and set analysis times
    timeInfo = nlast_info()
    
    tbeg = kwargs.get('tbeg',0)
    tend = kwargs.get('tend',timeInfo['nlast'])
    dt = tend-tbeg+1
    

    #fetch grid info and set analysis domain
    data = pp.pload(0)
    
    data = too.setCompRange(data,**kwargs)
    
    L = np.zeros((dt,data.n1))
    
    for tt in range(0,dt):
        data = pp.pload(tt+tbeg)
        data = too.setCompRange(data,**kwargs)
        
        L[tt,:] = rossbyL(data,gamma)

    return L


def rhovrAverage(tbeg,tend,**kwargs):
    '''
    calculate non-weighted  averages for radial mass flux
    
    Parameters:
    -----------
     tbeg : beginning of time average
     tend : end of time average
     kwargs : arguments for clipping computational range with setCompRange

    Returns:
    --------
     dictionary with numpy arrays as objects. Keys are:
             rhovrMer : radial,azimuth and time average of rho*vx1
             rhovrRad : meridional, azimuth and time average of rho*vx1
    '''
    
    data = pp.pload(tbeg)
    data = too.setCompRange(data,**kwargs)
    
    rhovrMer = np.zeros(data.n2)
    rhovrRad = np.zeros(data.n1)
    
    dt = tend-tbeg+1
    for t in range(tbeg,tend+1):
        if t > tbeg:
            data = pp.pload(t)
            data = too.setCompRange(data,**kwargs)
        rhovrMer += np.average(data.rho*data.vx1,axis=(0,2))
        rhovrRad += np.average(data.rho*data.vx1,axis=(1,2))
    
    rhovrMer /= dt
    rhovrRad /= dt
    
    return {'rhovrMer':rhovrMer,'rhovrRad':rhovrRad}

def vrmsAzAverage(data,rhow=True,vphimean=None):
    '''
    calculate the azimuthal average of the rms velocity
    
    Parameters:
    -----------
        data : input PLUTO dataset
        rhow : use density weighting (optional, default: True)
        vphimean : average phi-velocity profile (optional; default: None, calculates phi-average)
        
    Returns:
    --------
        vrms : 2d - array containing vrms(R,Z) 
    '''

    dx3 = data.dx3[None,None,:]
    
    if (vphimean is None):
        vphimean = np.sum(data.vx3*dx3,axis=2)/np.sum(data.dx3)
    
    if rhow:
        vsqmean = np.sum(data.rho*(data.vx1*data.vx1+data.vx2*data.vx2+(data.vx3-vphimean[:,:,None])*(data.vx3-vphimean[:,:,None]))
        *dx3,axis=2)/np.sum(data.rho*dx3,axis=2)
    else:
        vsqmean = np.sum((data.vx1*data.vx1+data.vx2*data.vx2+(data.vx3-vphimean[:,:,None])*(data.vx3-vphimean[:,:,None]))*dx3,axis=2)/np.sum(data.x3)

    return np.sqrt(vsqmean)

def vrmsAzMerAverage(data,rhow=True,vphimean=None):
    '''
    calcuclate the azimuthal and meridional average of the rms-velocity
    
    Parameters:
    -----------
        data : input PLUTO dataset
        rhow : use density weighting (optional, default: True)
        vphimean : average phi-velocity profile (optional; default: None, calculates phi-average)
        
    Returns:
    --------
        vrms : 1d - array containing vrms(R) 
    '''

    dx3 = data.dx3[None,None,:]
    
    if (vphimean is None):
        vphimean = np.sum(data.vx3*dx3,axis=2)/np.sum(data.dx3)
    
    if rhow:
        vsqmean = np.sum(data.rho*(data.vx1*data.vx1+data.vx2*data.vx2+(data.vx3-vphimean[:,:,None])*(data.vx3-vphimean[:,:,None]))*dx3
        *(np.sin(data.x2)*data.dx2)[None,:,None],axis=(1,2))/np.sum(data.rho*(np.sin(data.x2)*data.dx2)[None,:,None]*dx3)
    else:
        vsqmean = np.sum((data.vx1*data.vx1+data.vx2*data.vx2+(data.vx3-vphimean[:,:,None])*(data.vx3-vphimean[:,:,None]))*(np.sin(data.x2)*data.dx2)[None,:,None]*dx3,axis=(1,2))/np.sum((np.sin(data.x2)*data.dx2)[None,:,None]*dx3,axis=(1,2))

    return np.sqrt(vsqmean)

def vrmsAzRadAverage(data,rhow=True,vphimean=None):
    '''
    calculate the azimuthal and radial average of the rms-velocity
    
    Parameters:
    -----------
        data : input PLUTO dataset
        rhow : use density weighting (optional, default: True)
        vphimean : average phi-velocity profile (optional; default: None, calculates phi-average)
        
    Returns:
    --------
        vrms : 1d - array containing vrms(Z) 
    '''

    dx3 = data.dx3[None,None,:]
    
    if (vphimean is None):
        vphimean = np.sum(data.vx3*dx3,axis=2)/np.sum(data.dx3)
        
   
    if rhow:
        vsqmean = np.sum(data.rho*(data.vx1*data.vx1+data.vx2*data.vx2+(data.vx3-vphimean[:,:,None])*(data.vx3-vphimean[:,:,None]))
        *(data.x1*data.dx1)[:,None,None]*dx3,axis=(0,2))/np.sum(data.rho*(data.x1*data.dx1)[:,None,None]*dx3,axis=(0,2))
    else:
        vsqmean = np.sum((data.vx1*data.vx1+data.vx2*data.vx2+(data.vx3-vphimean[:,:,None])*(data.vx3-vphimean[:,:,None]))
        *(data.x1*data.dx1)[:,None,None]*dx3,axis=(0,2))/np.sum((data.x1*data.dx1)[:,None,None]*dx3,axis=(0,2))
    
    return np.sqrt(vsqmean)


def vrmsAllAverage(data,rhow=True,vphimean=None):
    '''
    calculates the average of the rms-velocity over the whole simulation domain
    
    Parameters:
    ----------
    data :  The pyPLUTO  dataset to evaluate
    rhow :  Weather to use density weighting, default is true.
    vphimean : average phi-velocity profile (optional; default: None, calculates phi-average)
    
    Returns:
    -------
    rms velocity values  
    
    '''

    dx3 = data.dx3[None,None,:]
    if (vphimean is None):
        vphimean = np.sum(data.vx3*dx3,axis=2)/np.sum(data.dx3)
      
    if rhow:
        vsqmean = np.sum(data.rho*(data.vx1*data.vx1+data.vx2*data.vx2+(data.vx3-vphimean[:,:,None])*(data.vx3-vphimean[:,:,None]))
        *(data.x1*data.x1*data.dx1)[:,None,None]*(np.sin(data.x2)*data.dx2)[None,:,None]*dx3)/np.sum(data.rho
        *(data.x1*data.x1*data.dx1)[:,None,None]*(np.sin(data.x2)*data.dx2)[None,:,None]*dx3)
    else:
        vsqmean = np.sum((data.vx1*data.vx1+data.vx2*data.vx2+(data.vx3-vphimean[:,:,None])*(data.vx3-vphimean[:,:,None]))
        *(data.x1*data.x1*data.dx1)[:,None,None]*(np.sin(data.x2)*data.dx2)[None,:,None]*dx3)/np.sum(
        (data.x1*data.x1*data.dx1)[:,None,None]*(np.sin(data.x2)*data.dx2)[None,:,None]*dx3)
    
    return np.sqrt(vsqmean)

def vrmsAverage(**kwargs):
    '''
    wrapper calculating rhe rms-velocity timeseries and various time-averages
    
    Parameters:
    -----------
    tbeg: timestep to begin the vrms analysis, default is 0
    tend: timestep to end the vrms analysis, default is nlast
    rhow: weight the velocity with density, default is True
    vphimean : average phi-velocity profile (optional; default: None,
                options: value - 3d array containing vphimean
                         time - use phi,time average over last 100 orbits, 
                         None - always use phi-average of current timestep)
    **kwargs: all other kwargs are handed over to setCompRange
    
    returns:
    ----
    vrmsdict : dictionary containing averaged quantities
    
        vrmsTs: Volume averaged vrms vs time
        vrmsMer: R,phi,time averaged vrms vs theta
        vrmsR: theta,phi,time averaged vrms vr r
    
    '''
    
    rhow = kwargs.pop('rhow',True)
    vphimean = kwargs.pop('vphimean',None)
    
    #fetch time info and set analysis times
    timeInfo = nlast_info()
    
    tbeg = kwargs.pop('tbeg',0)
    tend = kwargs.pop('tend',timeInfo['nlast'])
    dt = tend-tbeg+1
    
    #fetch grid info and set analysis domain
    data = pp.pload(0)
    
    data = too.setCompRange(data,**kwargs)
    
    #define & initialize arrays
    vrmsTser= np.zeros(dt)
    vrmsAzRTav = np.zeros(data.n2)
    vrmsAzMerTav = np.zeros(data.n1)

    #calculate vphimean if necessary
    if (vphimean == 'time'):
        vphimean = np.zeros((data.n1,data.n2))
        if dt < 100:
            tbegphi = tbeg
        else:
            tbegphi = tend - 100
        for tt in range(tbegphi,tend):
            data = pp.pload(tt)
            data = too.setCompRange(data,**kwargs)
            vphimean += np.sum(data.vx3*data.dx3[None,None,:],axis=2)/np.sum(data.dx3)
        vphimean /= (tend-tbegphi)
    
    for tt in range(0,dt):
        data = pp.pload(tt+tbeg)
        data = too.setCompRange(data,**kwargs)
        vrmsTser[tt] = vrmsAllAverage(data,rhow,vphimean)
        vrmsAzRTav += vrmsAzRadAverage(data,rhow,vphimean)
        vrmsAzMerTav +=vrmsAzMerAverage(data,rhow,vphimean)
        
    vrmsAzRTav /= dt
    vrmsAzMerTav /= dt
    
    vrmsdict = {'vrmsTs':vrmsTser,'vrmsMer':vrmsAzRTav,'vrmsR':vrmsAzMerTav}
    
    return vrmsdict

def vrAverage(tbeg,tend,cr_kwargs,rhoAverage=True):
    '''
    calculate the averaged radial velocity quantities
    
    parameters:
    ----------
     tbeg:  the begin timestep for the time average
     tend:  the end timestep for the time average
     cr_kwargs: the kwargs for setComputationalRange()
     rhoAverage: average the velocity values with density. Defaulte is True.
    
     returns:
     -------
     vrdict: dictionay containing averaged quantites.
             Keys: 'vrMer'    -- radial velocity vs x2
                   'vrrmsMer' -- radial rms-velocity vs x2
                   'vrsqMer'  -- radial velocity squared vs x2
                   'vrRad'    -- radial velocity vs x1
                   'vrrmsRad' -- radial rms-velocity vs x1
                   'vrsqRad'  -- radial velocity squared vs x1
    '''
    
    grid = pp.pload(0)
    grid = too.setCompRange(grid,**cr_kwargs)
    timeInfo = nlast_info()
    if tbeg is None: tbeg = 0
    if tend is None: tend =  timeInfo['nlast']
    dt = tend - tbeg+1
    
    vrAzRadTav = np.zeros(grid.n2)
    vrAzRadTavrms = np.zeros(grid.n2)
    vrsqAzRadTav = np.zeros(grid.n2)
    
    vrAzMerTav = np.zeros(grid.n1)
    vrAzMerTavrms = np.zeros(grid.n1)
    vrsqAzMerTav = np.zeros(grid.n1)


    for t in range(tbeg,tend+1):
        data = pp.pload(t)
        data = too.setCompRange(data,**cr_kwargs)
        
        if rhoAverage:
            weights = data.rho
        else:
            weights = None
        
        vrAzRadTav += np.average(data.vx1,axis=(0,2),weights=weights) 
        vrAzRadTavrms += np.sqrt(np.average(data.vx1*data.vx1,axis=(0,2),weights=weights))
        vrsqAzRadTav += np.average(data.vx1*data.vx1,axis=(0,2),weights=weights)

        vrAzMerTav += np.average(data.vx1,axis=(1,2),weights=weights) 
        vrAzMerTavrms += np.sqrt(np.average(data.vx1*data.vx1,axis=(1,2),weights=weights))
        vrsqAzMerTav += np.average(data.vx1*data.vx1,axis=(1,2),weights=weights)
        
    vrAzRadTav /= dt
    vrAzRadTavrms /= dt
    vrsqAzRadTav /= dt
    
    vrAzMerTav /= dt
    vrAzMerTavrms /= dt
    vrsqAzMerTav /= dt
    
    vrdict = {'vrMer':vrAzRadTav ,'vrrmsMer':vrAzRadTavrms ,'vrsqMer':vrsqAzRadTav 
              ,'vrRad':vrAzMerTav , 'vrrmsRad':vrAzMerTavrms ,'vrsqRad':vrsqAzMerTav}
    return vrdict

def vthAverage(cr_kwargs,tbeg=None,tend=None,rhoAverage=False):
    '''
    calculate multiple averages for meridional  velocity
    
    parameters:
    ----------
     tbeg:  the begin timestep for the time average
     tend:  the end timestep for the time average
     cr_kwargs: the kwargs for setComputationalRange()
     rhoAverage: average the velocity values with density. Defaulte is False.
    
     returns:
     -------
     
    '''
    
    grid = pp.pload(0)
    grid = too.setCompRange(grid,**cr_kwargs)
    timeInfo = nlast_info()
    if tbeg is None: tbeg = 0
    if tend is None: tend =  timeInfo['nlast']
    dt = tend - tbeg+1
    
    vzAzRadTav = np.zeros(grid.n2)
    vzAzRadTavrms = np.zeros(grid.n2)
    vzsqAzRadTav = np.zeros(grid.n2)
    
    vzAzMerTav = np.zeros(grid.n1)
    vzAzMerTavrms = np.zeros(grid.n1)
    vzsqAzMerTav = np.zeros(grid.n1)
    
    for t in range(tbeg,tend+1):
        data = pp.pload(t)
        data = too.setCompRange(data,**cr_kwargs)
    
        if rhoAverage:
            weights = data.rho
        else:
            weights = None
        
        vzAzRadTav += np.average(data.vx2,axis=(0,2),weights=weights)
        vzAzRadTavrms += np.sqrt(np.average(data.vx2*data.vx2,axis=(0,2),weights=weights))
        vzsqAzRadTav += np.average(data.vx2*data.vx2,axis=(0,2),weights=weights)
        
        vzAzMerTav += np.average(data.vx2,axis=(1,2),weights=weights)
        vzAzMerTavrms += np.sqrt(np.average(data.vx2*data.vx2,axis=(1,2),weights=weights))
        vzsqAzMerTav += np.average(data.vx2*data.vx2,axis=(1,2),weights=weights)

    vzAzRadTav /= dt
    vzAzRadTavrms /= dt
    vzsqAzRadTav /= dt
    
    vzAzMerTav /= dt
    vzAzMerTavrms /= dt
    vzsqAzMerTav /= dt
    
    vthdict = {'vthMer':vzAzRadTav ,'vthrmsMer':vzAzRadTavrms ,'vthsqMer':vzsqAzRadTav 
              ,'vthRad':vzAzMerTav , 'vthrmsRad':vzAzMerTavrms ,'vthsqRad':vzsqAzMerTav}
    return vthdict


def ekinAzAverage(data):

    nx1 = getattr(data,'n1')
    nx2 = getattr(data,'n2')
    nx3 = getattr(data,'n3')
    nx3m1 = 1.0/(1.0*nx3)

    ekinmean = np.zeros((nx1,nx2))
    ekins = 0.0

    for i in data.irange:
        for j in data.jrange:
            ekinmean[i,j] = 0.5*np.sum(data.rho[i,j,:]*(data.vx1[i,j,:]*data.vx1[i,j,:]+data.vx2[i,j,:]*data.vx2[i,j,:]))*nx3m1
    
    return ekinmean

def ekinAzMerAverage(data):
    
    nx1 = getattr(data,'n1')
    nx2 = getattr(data,'n2')
    nx3 = getattr(data,'n3')
    nx3m1 = 1.0/(1.0*nx3)
    nx2m1 = 1.0/(1.0*nx2)

    ekin1d = np.zeros(nx1)
    ekin2d = ekinAzAverage(data)
    
    for i in data.irange:
        ekin1d = 0.5*np.sum(data.rho[i,:,:]*(data.vx1[i,:,:]*data.vx1[i,:,:]+data.vx2[i,:,:]*data.vx2[i,:,:]))*nx3m1*nx2m1
    
    return ekin1d

def ekinAzRadAverage(data):
    
    nx1 = getattr(data,'n1')
    nx2 = getattr(data,'n2')
    nx3 = getattr(data,'n3')
    nx3m1 = 1.0/(1.0*nx3)
    nx1m1 = 1.0/(1.0*nx1)

    ekin1d = np.zeros(nx2)
    
    for j in data.jrange:
        ekin1d = 0.5*np.sum(data.rho[:,j,:]*(data.vx1[:,j,:]*data.vx1[:,j,:]+data.vx2[:,j,:]*data.vx2[:,j,:]))*nx3m1*nx1m1
    
    return ekin1d


def ekinAllAverage(data):

    nx1 = getattr(data,'n1')
    nx1m1 = 1.0/(1.0*nx1)
    
    ekin1d = ekinAzMerAverage(data)
    ekinTot = np.sum(ekin1d)*nx1m1
    
    return ekinTot

def ekinTAllAverage(beg,end):

    T = 1.0*(end-beg+1)
    ekinTav = 0.0
    for i in range(beg,end+1):
        data = pp.pload(i)
        ekinTav += ekinAllAverage(data)
    
    return ekinTav/T


def rhoAzAverage(data):

    nx1 = getattr(data,'n1')
    nx2 = getattr(data,'n2')
    nx3 = getattr(data,'n3')
    nx3m1 = 1.0/(1.0*nx3)

    rhomean = np.zeros((nx1,nx2))
    
    for i in data.irange:
        for j in data.jrange:
            rhomean[i,j] = np.sum(data.rho[i,j,:])*nx3m1
    
    return rhomean

def rhoAzMerAverage(data):
    
    nx1 = getattr(data,'n1')
    nx2 = getattr(data,'n2')
    nx3 = getattr(data,'n3')
    nx3m1 = 1.0/(1.0*nx3)
    nx2m1 = 1.0/(1.0*nx2)

    rho1d = np.zeros(nx1)
    
    for i in data.irange:
        rho1d = np.sum(data.rho[i,:,:])*nx2m1*nx3m1
    
    return rho1d

def rhoAzRadAverage(data):
    
    nx1 = getattr(data,'n1')
    nx2 = getattr(data,'n2')
    nx3 = getattr(data,'n3')
    nx3m1 = 1.0/(1.0*nx3)
    nx1m1 = 1.0/(1.0*nx1)

    rho1d = np.zeros(nx2)
    
    for j in data.jrange:
        rho1d = np.sum(data.rho[:,j,:])*nx1m1*nx3m1
    
    return rho1d


def rhoAllAverage(data):

    nx1 = getattr(data,'n1')
    nx1m1 = 1.0/(1.0*nx1)
    
    rho1d = rhoAzMerAverage(data)
    rhoTot = np.sum(rho1d)*nx1m1
    
    return rhoTot

def rhoTAllAverage(beg,end):

    T = 1.0*(end-beg+1)
    rhoTav = 0.0
    for i in range(beg,end+1):
        data = pp.pload(i)
        rhoTav += rhoAllAverage(data)
    
    return rhoTav/T

def dumpToFile(dataset,filename):
    with open(filename,'w') as file:
        json.dump(dataset.tolist(),file)
