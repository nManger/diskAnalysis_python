import pyPLUTO as pp
import numpy as np
import matplotlib.pyplot as plt
import math as m
import tools3d as too
from scipy.signal import savgol_filter,convolve


def vortVal(data, **kwargs):
    
    vert = kwargs.pop('vertical',False)
    
    x1 = data.x1[:,None,None]
    x2 = data.x2[None,:,None]
    x3 = data.x3[None,None,:]
    
    if (vert==True):
      #  k= kwargs.pop('x3cut',int(round(data.n3*0.5)))

        vort = np.zeros((data.n1,data.n2,data.n3))
        #print vort.shape
#         for i in range(1,data.n1-1):
#             for j in range(0,data.n2):
#                 for k in range(1,data.n3-1):
#                     vort[i,j,k] = ((data.vx3[i+1,j,k]*data.x1[i+1]*np.sin(data.x2[j])-data.vx3[i-1,j,k]*data.x1[i-1]*np.sin(data.x2[j]))
#                                    /(data.x1[i+1]*np.sin(data.x2[j])-data.x1[i-1]*np.sin(data.x2[j]))/(data.x1[i]*np.sin(data.x2[j]))
#                                    -(data.vx1[i,j,k+1]*np.sin(data.x2[j])+data.vx2[i,j,k+1]*np.cos(data.x2[j])
#                                      -data.vx1[i,j,k-1]*np.sin(data.x2[j])-data.vx2[i,j,k-1]*np.cos(data.x2[j]))/(data.x3[k+1]-data.x3[k-1])
#                                    /(data.x1[i]*np.sin(data.x2[j])))/(data.vx3[i,j,k]/data.x1[i]/np.sin(data.x2[j]))
#                 vort[i,j,0]=vort[i,j,1]
#                 vort[i,j,data.n3-1]=vort[i,j,data.n3-2]
                    
        vort[1:-1,:,1:-1] = ((data.vx3[2:,:,1:-1]*x1[2:,:,:]*np.sin(x2)-data.vx3[0:-2,:,1:-1]*x1[0:-2,:,:]*np.sin(x2))
                             /(x1[2:,:,:]*np.sin(x2)-x1[0:-2,:,:]*np.sin(x2))/(x1[1:-1,:,:]*np.sin(x2))
                             -(data.vx1[1:-1,:,2:]*np.sin(x2)+data.vx2[1:-1,:,2:]*np.cos(x2)
                               -data.vx1[1:-1,:,0:-2]*np.sin(x2)-data.vx2[1:-1,:,0:-2]*np.cos(x2))
                               /(x3[:,:,2:]-x3[:,:,0:-2])/(x1[1:-1,:,:]*np.sin(x2)))/(data.vx3[1:-1,:,1:-1]/x1[1:-1,:,:]/np.sin(x2))
        
        vort[0,:,:] = vort[1,:,:]
        vort[-1,:,:] = vort[-2,:,:]
        vort[:,:,0] = vort[:,:,1]
        vort[:,:,-1] = vort[:,:,-2]
    
    else:
        vort = np.zeros((data.n1,data.n3))
        j= kwargs.pop('x2cut',int(round(data.n2*0.5)))
#         for i in range(1,data.n1-1):
#             for k in range(1,data.n3-1):
            #vorticity in z direction
#                 vort[i,k] = ((data.vx3[i+1,j,k]*data.x1[i+1]-data.vx3[i-1,j,k]*data.x1[i-1])/(data.x1[i+1]-data.x1[i-1])/(data.x1[i])-(data.vx1[i,j,k+1]-data.vx1[i,j,k-1])/(data.x3[k+1]-data.x3[k-1])/(data.x1[i]))/(data.vx3[i,j,k]/data.x1[i])
        
        vort[1:-1,1:-1] = ((data.vx3[2:,j,1:-1]*x1[2:,0,:]-data.vx3[0:-2,j,1:-1]*x1[:-2,0,:])/(x1[2:,0,:]-x1[:-2,0,:])/(x1[1:-1,0,:])
                            -(data.vx1[1:-1,j,2:]-data.vx1[1:-1,j,:-2])/(x3[:,0,2:]-x3[:,0,:-2])/(x1[1:-1,0,:])
                            )/(data.vx3[1:-1,j,1:-1]/x1[1:-1,0,:])
           
        vort[:,0]=vort[:,1]
        vort[:,-1]=vort[:,-2]
        vort[0,:]=vort[1,:]
        vort[-1,:] = vort[-2,:]
    
    return vort

def vortRTheta(data,**kwargs):
    
    r,th = np.meshgrid(data.x1r,data.x2r,indexing='ij')

    xx = r*np.sin(th)
    yy = r*np.cos(th)
    
    
    k= kwargs.pop('x3cut',int(round(data.n3*0.5)))

    vort = np.zeros((data.n1,data.n2))
    for i in range(1,data.n1-1):
        for j in range(0,data.n2):
            #vort[i,j] = ((data.vx1[i,j,k+1]-data.vx1[i,j,k-1])/(data.x3[k+1]-data.x3[k-1])/(data.x1[i]*m.sin(data.x2[j]))-(data.vx3[i+1,j,k]*data.x1[i+1]-data.vx3[i-1,j,k]*data.x1[i-1])/(data.x1[i+1]-data.x1[i-1])/(data.x1[i]))/(data.vx3[i,j,k]/data.x1[i])
            vort[i,j] = ((data.vx3[i+1,j,k]*data.x1[i+1]-data.vx3[i-1,j,k]*data.x1[i-1])/(data.x1[i+1]-data.x1[i-1])/(data.x1[i])-(data.vx1[i,j,k+1]-data.vx1[i,j,k-1])/(data.x3[k+1]-data.x3[k-1])/(data.x1[i]))/(data.vx3[i,j,k]/data.x1[i])
            
    vort[0,:] = vort[1,:]
    vort[data.n1-1,:] = vort[data.n1-2,:]
    
    f1=plt.figure(kwargs.pop('fignum',1),figsize=kwargs.pop('figsize',[6,5]),dpi=80,facecolor='w',edgecolor='k')
    ax1=f1.add_subplot(111)
    ax1.set_aspect(kwargs.pop('aspect','auto'))
    #ax1.axis([np.min(xx),np.max(xx),np.min(yy),np.max(yy)])
    
    plt.pcolormesh(xx,yy,vort,vmin=0.0,vmax=1.0,cmap = kwargs.pop('cmap','Blues_r'))
    plt.xlabel('R',size='large')
    plt.ylabel('z',size='large')
    plt.ylim(-0.7,0.7)
    plt.colorbar()
    plt.tight_layout()

def vortRPhi(data,**kwargs):
    
    vort = np.zeros((data.n1,data.n3))
    j= kwargs.pop('x2cut',int(round(data.n2*0.5)))
    for i in range(1,data.n1-1):
        for k in range(1,data.n3-1):
            #vorticity in theta direction
            #vort[i,k] = ((data.vx1[i,j,k+1]-data.vx1[i,j,k-1])/(data.x3[k+1]-data.x3[k-1])/(data.x1[i]*m.sin(data.x2[j]))-(data.vx3[i+1,j,k]*data.x1[i+1]-data.vx3[i-1,j,k]*data.x1[i-1])/(data.x1[i+1]-data.x1[i-1])/(data.x1[i]))/(data.vx3[i,j,k]/data.x1[i])
            #vorticity in z direction
            vort[i,k] = ((data.vx3[i+1,j,k]*data.x1[i+1]-data.vx3[i-1,j,k]*data.x1[i-1])/(data.x1[i+1]-data.x1[i-1])/(data.x1[i])-(data.vx1[i,j,k+1]-data.vx1[i,j,k-1])/(data.x3[k+1]-data.x3[k-1])/(data.x1[i]))/(data.vx3[i,j,k]/data.x1[i])
            #if vort[i,k] < -1.0: vort[i,k]=-1.0
            #if vort[i,k] > 1.0: vort[i,k]=1.0
        vort[i,0]=vort[0,1]
        vort[i,data.n3-1]=vort[i,data.n3-2]
    vort[0,:]=vort[1,:]
    vort[data.n1-1,:] = vort[data.n1-2,:]
    
    im= kwargs.pop('contour',False)
    if (im==True):
        levels=np.arange(0.0,0.5+0.1,0.05)
        x,y = np.meshgrid(data.x1,data.x3,indexing='ij')
        plt.figure(kwargs.pop('fignum',1),figsize=[10,10],dpi=80,facecolor='w',edgecolor='k')
        cs=plt.contour(x,y,vort,levels,cmap=kwargs.pop('cmap','Blues_r'),linewidths=1.0)
        cb = plt.colorbar(cs)
        cb.lines[0].set_linewidth(4.0)
    else:
         I = pp.Image()
         aspect = kwargs.pop('aspect','auto')
         cmap = kwargs.pop('cmap','Blues_r')
         figsize = kwargs.pop('figsize',[5,5])
         I.pldisplay(data,vort,x1=data.x1,x2=data.x3,vmin=0.0,vmax=0.5,cbar=(True,'vertical'),label1='R',label2=r'$\phi$',aspect=aspect,cmap=cmap,figsize=figsize,size='large',fignum=kwargs.pop('fignum',1))
   # plt.waitforbuttonpress()
    return vort

def vortXY(data,**kwargs):
    vort = np.zeros((data.n1,data.n3))
    j= kwargs.pop('x2cut',int(round(data.n2*0.5)))
    for i in range(1,data.n1-1):
        for k in range(1,data.n3-1):
            #vorticity in theta direction
            #vort[i,k] = ((data.vx1[i,j,k+1]-data.vx1[i,j,k-1])/(data.x3[k+1]-data.x3[k-1])/(data.x1[i]*m.sin(data.x2[j]))-(data.vx3[i+1,j,k]*data.x1[i+1]-data.vx3[i-1,j,k]*data.x1[i-1])/(data.x1[i+1]-data.x1[i-1])/(data.x1[i]))/(data.vx3[i,j,k]/data.x1[i])
            #vorticity in z direction
            vort[i,k] = ((data.vx3[i+1,j,k]*data.x1[i+1]-data.vx3[i-1,j,k]*data.x1[i-1])/(data.x1[i+1]-data.x1[i-1])/(data.x1[i])-(data.vx1[i,j,k+1]-data.vx1[i,j,k-1])/(data.x3[k+1]-data.x3[k-1])/(data.x1[i]))/(data.vx3[i,j,k]/data.x1[i])
            #if vort[i,k] < -1.0: vort[i,k]=-1.0
            #if vort[i,k] > 1.0: vort[i,k]=1.0
        vort[i,0]=vort[0,1]
        vort[i,data.n3-1]=vort[i,data.n3-2]
    vort[0,:]=vort[1,:]
    vort[data.n1-1,:] = vort[data.n1-2,:]

    f1=plt.figure(kwargs.pop('fignum',1),dpi=80,facecolor='w',edgecolor='k')
    ax1=f1.add_subplot(111)
    ax1.set_aspect(kwargs.pop('aspect','equal'))
    
    xx,yy = too.getCartGrid(data,rphi=True)

    plt.pcolormesh(xx,yy,vort,vmin=0.0,vmax=0.5,cmap = kwargs.pop('cmap','Blues_r'))
    plt.xlabel('X',size='large')
    plt.ylabel('Y',size='large')
    plt.colorbar()


def vortThetaPhi(data,**kwargs):
    
    vort = np.zeros((data.n2,data.n3))
    i= kwargs.pop('x1cut',int(round(data.n1*0.5)))
    
    for j in range(0,data.n2):
        for k in range(1,data.n3-1):
           # vort[j,k] = ((data.vx1[i,j,k+1]-data.vx1[i,j,k-1])/(data.x3[k+1]-data.x3[k-1])/(data.x1[i]*m.sin(data.x2[j]))-(data.vx3[i+1,j,k]*data.x1[i+1]-data.vx3[i-1,j,k]*data.x1[i-1])/(data.x1[i+1]-data.x1[i-1])/(data.x1[i]))/(data.vx3[i,j,k]/data.x1[i])
            vort[j,k] = ((data.vx3[i+1,j,k]*data.x1[i+1]-data.vx3[i-1,j,k]*data.x1[i-1])/(data.x1[i+1]-data.x1[i-1])/(data.x1[i])-(data.vx1[i,j,k+1]-data.vx1[i,j,k-1])/(data.x3[k+1]-data.x3[k-1])/(data.x1[i]))/(data.vx3[i,j,k]/data.x1[i])
        
        vort[j,0] = vort[j,1]
        vort[j,data.n3-1] = vort[j,data.n3-2]

    cmap=kwargs.pop('cmap','Blues_r')
    I=pp.Image()
    I.pldisplay(data,np.transpose(vort),x1=data.x3,x2=data.x2,vmin=-0.5,vmax=1.5,cbar=(True,'vertical'),label1=r'$\phi$',label2=r'$\theta$',aspect='auto',cmap=cmap,figsize=[5,5],size='large', **kwargs)
    

def vortRPhiAv(data,**kwargs):
    
    vort = np.zeros((data.n1,data.n3))
    
    thmin = kwargs.pop('thmin',0)
    thmax = kwargs.pop('thmax',data.n2-1)
    nth= thmax-thmin+1
    
    for i in range(1,data.n1-1):
        for k in range(1,data.n3-1):
            for j in range(thmin,thmax+1):
                #vorticity in z direction
                vort[i,k] += ((data.vx3[i+1,j,k]*data.x1[i+1]-data.vx3[i-1,j,k]*data.x1[i-1])/(data.x1[i+1]-data.x1[i-1])/(data.x1[i])-(data.vx1[i,j,k+1]-data.vx1[i,j,k-1])/(data.x3[k+1]-data.x3[k-1])/(data.x1[i]))/(data.vx3[i,j,k]/data.x1[i])/nth
        vort[i,0]=vort[i,1]
        vort[i,data.n3-1]=vort[i,data.n3-2]
    vort[0,:]=vort[1,:]
    vort[data.n1-1,:] = vort[data.n1-2,:]

    I = pp.Image()
    I.pldisplay(data,vort,x1=data.x1,x2=data.x3,vmin=-0.0,vmax=0.5,cbar=(True,'vertical'),label1='R',label2=r'$\phi$',aspect='auto',cmap=kwargs.pop('cmap','Blues_r'),figsize=[5,5],size='large',fignum=kwargs.pop('fignum',1))
    return vort

def vortRav(data,**kwargs):
    
    rmin = kwargs.pop('rmin',0)
    rmax = kwargs.pop('rmax',data.n1-1)
    thmin = kwargs.pop('thmin',0)
    thmax = kwargs.pop('thmax',data.n2-1)
    
    nr  = rmax-rmin+1
    nth = thmax-thmin+1

    vort = np.zeros(nr)
    
    for i in range(rmin+1,rmax):
        ii = i-rmin
        for k in range(1,data.n3-1):
            for j in range(thmin,thmax+1):
                #vorticity in z direction
                vort[ii] += ((data.vx3[i+1,j,k]*data.x1[i+1]-data.vx3[i-1,j,k]*data.x1[i-1])/(data.x1[i+1]-data.x1[i-1])/(data.x1[i])-(data.vx1[i,j,k+1]-data.vx1[i,j,k-1])/(data.x3[k+1]-data.x3[k-1])/(data.x1[i]))/(data.vx3[i,j,k]/data.x1[i])/nth
        vort[ii]/=(data.n3-2)
    vort[0]=vort[1]
    vort[nr-1] = vort[nr-2]

    plt.figure(kwargs.pop('fignum',1),figsize=[5,5],dpi=80,facecolor='w',edgecolor='k')
    plt.plot(data.x1[rmin:rmax+1],vort)
    plt.xlabel('R',size='large')
    plt.ylabel(r'$\omega_z/\Omega$')
    
    #return vort

def kappaSqR(data,**kwargs):
    
    j=kwargs.pop('x2cut',int(round(data.n2/2.0)))
    k=kwargs.pop('x3cut',int(round(data.n3/2.0)))

    kapsq=np.zeros(data.n1)
    omgsq = data.vx3[:,j,k]*data.vx3[:,j,k]/(data.x1*data.x1)
    
    for i in range(1,data.n1-1):
        kapsq[i] = 1.0/m.pow(data.x1[i],3.0)*(m.pow(data.x1[i+1],4.0)*omgsq[i+1]-m.pow(data.x1[i-1],4.0)*omgsq[i-1])/(data.x1[i+1]-data.x1[i-1])

    kapsq[0]=kapsq[1]
    kapsq[data.n1-1]=kapsq[data.n1-2]

    return kapsq/omgsq

def boxFilter(data,n,axis):
    '''
    This is a simple box filter used in the time trace of vortices.

    parameters:
    ----------
    data :   2d numpy array containing the data
       n :   width of the filter kernel in grid cells
    axis :   axis to apply the filter to

    return:
    ------
        the smoothed numpy array with the same size as data
    '''
    
    dshape = data.shape
    conv = np.zeros(dshape)
    win = np.ones(n)/n

    if (axis ==0):
        dataIn = np.zeros((dshape[0]+n-1,dshape[1]))
        for i in range(0,dshape[0]+n-1):
            for j in range(0,dshape[1]):
                if (i < n/2):
                    dataIn[i,j] = data[0,j]
                else:
                    if(i > dshape[0]+n/2-1):
                        dataIn[i,j] = data[-1,j]
                    else:
                        dataIn[i,j] = data[i-n/2,j]
                
        for i in range(0,dshape[1]):
            conv[:,i] = convolve(dataIn[:,i],win,mode='valid')
    else:
        dataIn = np.zeros((dshape[0],dshape[1]+n-1))
        for i in range(0,dshape[0]):
            for j in range(0,dshape[1]+n-1):
                if (j < n/2):
                    dataIn[i,j] = data[i,dshape[1]-n/2+j]
                else:
                    if(j > dshape[1]+n/2-1):
                        dataIn[i,j] = data[i,j-dshape[1]-n/2]
                    else:
                        dataIn[i,j] = data[i,j-n/2]
        for i in range(0,dshape[0]):
            conv[i] = convolve(dataIn[i,:],win,mode='valid')
        
    return conv


def vortTimeAnalysis(**kwargs):
    
    timeInfo = pp.nlast_info()    
    tnbeg = kwargs.pop('tbeg',0)
    tnend = kwargs.pop('tend',timeInfo['nlast'])
    dtn = tnend - tnbeg+1
    
    data = pp.pload(0)

    dnfphi = kwargs.pop('dnphi',200)        #default filter width for vortices in phi direction   
    dnfr = kwargs.pop('dnr',20)        #default filter width for vortices in phi direction   
    #hor = kwargs.pop('hor',0.1)
    #hmax = kwargs.pop('hmax',2.0)
    
    #x2min = data.x2[m.floor(data.n2/2.0)] - hor*hmax
    #x2max = data.x2[m.floor(data.n2/2.0)] + hor*hmax
    
    # for j in xrange(data.n2):
#         if data.x2[j] >= x2min:
#             n2min= j
#             break
#     for j in xrange(data.n2-1,0,-1):
#         if data.x2[j] <= x2max:
#             n2max= j
#             break
#     nth = n2max-n2min+1

    vortRtime=np.zeros((dtn,data.n1))
    
    for t in range(tnbeg,tnend+1):
        tt = t-tnbeg
        
        data = pp.pload(t)
        
        vort = vortVal(data,x2cut=data.n2/2)
        
#         vort = np.zeros((data.n1,data.n3))
#         
#         j = data.n2/2
#         for i in range(1,data.n1-1):
#             for k in range(1,data.n3-1):
#                #vorticity in z direction
#                 vort[i,k] = ((data.vx3[i+1,j,k]*data.x1[i+1]-data.vx3[i-1,j,k]*data.x1[i-1])/(data.x1[i+1]-data.x1[i-1])/(data.x1[i])
#                                -(data.vx1[i,j,k+1]-data.vx1[i,j,k-1])/(data.x3[k+1]-data.x3[k-1])/(data.x1[i]))/(data.vx3[i,j,k]/data.x1[i])
#                 vort[i,0]=vort[i,1]
#                 vort[i,data.n3-1]=vort[i,data.n3-2]
#             vort[0,:]=vort[1,:]
#             vort[data.n1-1,:] = vort[data.n1-2,:]
       

        if (data.n3 < dnfphi):   # check if filter would cover entire domain, if yes set fiter width to 1/2 domain
            dnfphi = int(data.n3/2)
        vort = boxFilter(np.clip(vort,0.0,0.5),dnfr,0) #filter in r direction
        vort = boxFilter(vort,dnfphi,1)              #filter in phi direction

#         vort = savgol_filter(vort,21,0,axis=1,mode='wrap') #filter in phi direction
#         vort = savgol_filter(vort,11,0,axis=0,mode='mirror') #filter in r direction

        phi_av = np.average(vort,axis=1)
#         r_av = np.average(vort,axis=0)
        
#         rmesh,phimesh = np.meshgrid(phi_av,r_av,indexing='ij')
        
#         vortDev = vort - rmesh - phimesh
         
        vort = vort - np.outer(phi_av,np.ones(data.n3))
         
        vortRtime[tt,:] = np.amin(vort,axis=1)
        
    return vortRtime
