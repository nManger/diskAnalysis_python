import numpy as np
import pyPLUTO as pp
import json
import copy

#def sph2cyl3D(data,**kwargs):
    
   # for k in data.krange:

def dumpToFile(dataset,filename):
    ''' Dump dataset to json formated file
    
    Parameters
    ----------
    dataset  : 2D numpy array to be saved
    filename : name of the file

    '''
    
    with open(filename,'w') as file:
        json.dump(dataset.tolist(),file)


def getCartGrid(data,**kwargs):
    ''' Generate cartesian grid from spherical
    
    Parameters
    ----------
    data  : data dictionary with spherical coordinate set
    rphi  : if r-phi (True) or r-theta (False) plane shall be transformed. Defaults to false.
    '''
    rphi=kwargs.pop('rphi',False)

    if (rphi == True):
        r,phi = np.meshgrid(data.x1r,data.x3r,indexing='ij')
        
        xx = r*np.cos(phi)
        yy = r*np.sin(phi)
        
    else:
        r,th = np.meshgrid(data.x1r,data.x2r,indexing='ij')

        xx = r*np.sin(th)
        yy = r*np.cos(th)

    return xx,yy
        

def getNrSpecies(data):
    ''' Get the type and number of species used in data.
    
    Parameters
    ----------
    data : pyPLUTO object

    Returns
    -------
    nspec : int
        number of species in the dataset
    stype : string
        either 'dsp' or 'trc', indicating the variable type used    
    '''

    nspec = 0
    stype = ''
    vars = data.vars
    if 'tr1' in vars:
        stype = 'trc'
        nspec = 1
    else:
        i=0
        stype = 'dsp'
        while True:
            key = 'dsp'+ str(i+1)
            if key in vars:
                i += 1
            else:
                break
        nspec = i
    return (nspec,stype)


def setCompRange(data,**kwargs):
    ''' Limit the grid range of data to a desired subset
    
    Parameters
    ----------
    data : pyload object
    
    Keyword args
    ------------
    either ximin/max or nimin/max should be given, for both xi and ni present ximin/max take precedence
    x1min, x2min : lower boundary x- value for each direction
    x1max, x2max : upper boundary x- value for each direction
    n1min, n2min : lower boundary index value for each direction (default is 0) 
    n1max, n2max : upper boundary index value for each direction (default is data.n[1,2,3])
    
    Returns
    -------
    data : pload object containting the desired subset
    '''
    
    dict = data.__dict__
    
    #get new x1 boundaries
    if 'x1min' in kwargs:
        x1min=kwargs.get('x1min')
        for i in xrange(data.n1):
            if data.x1[i] >= x1min:
                x1min= data.x1[i]
                n1min= i
                break
    else:
        n1min=kwargs.get('n1min',0)
        x1min=data.x1[n1min]
    
    if 'x1max' in kwargs:
        x1max=kwargs.get('x1max')
        for i in xrange(data.n1-1,0,-1):
            if data.x1[i] <= x1max:
                x1max= data.x1[i]
                n1max= i+1
                break
    else:
        n1max=kwargs.get('n1max',data.n1)
        x1max=data.x1[n1max-1]     
    
    #get new x2 boundaries
    if 'x2min' in kwargs:
        x2min=kwargs.get('x2min')
        for j in xrange(data.n2):
            if data.x2[j] >= x2min:
                x2min= data.x2[j]
                n2min= j
                break
    else:
        n2min=kwargs.get('n2min',0)
        x2min=data.x2[n2min]
    
    if 'x2max' in kwargs:
        x2max=kwargs.get('x2max')
        for j in xrange(data.n2-1,0,-1):
            if data.x2[j] <= x2max:
                x2max= data.x2[j]
                n2max= j+1
                break
    else:
        n2max=kwargs.get('n2max',data.n2)
        x2max=data.x2[n2max-1]


    #create new dictionary and copy non-changing information
    dictNew = {}
    dictNew['Slice'] = dict['Slice']
    dictNew['vars']  = dict['vars']
    dictNew['endianess']  = dict['endianess']
    dictNew['filetype']  = dict['filetype']
    dictNew['NStepStr']  = dict['NStepStr']
    dictNew['NStep']  = dict['NStep']
    dictNew['Dt']  = dict['Dt']
    dictNew['SimTime']  = dict['SimTime']
    dictNew['wdir']  = dict['wdir']
    dictNew['level']  = dict['level']
    dictNew['datatype']  = dict['datatype']
    dictNew['geometry']  = dict['geometry']
    
    dictNew['x1range'] = None
    dictNew['x2range'] = None
    dictNew['x3range'] = None
    
    dictNew['nshp'] = (n2max-n2min,n1max-n1min)
    
    #set new x1 direction
    dictNew['n1_tot'] = n1max-n1min
    dictNew['n1'] = n1max-n1min
    dictNew['irange']=[x for x in range(0,dictNew['n1'])]
    dictNew['x1'] = dict['x1'][n1min:n1max]
    dictNew['dx1'] = dict['dx1'][n1min:n1max]
    dictNew['x1r'] = dict['x1r'][n1min:n1max]

    #set new x2 direction
    dictNew['n2_tot'] = n2max-n2min
    dictNew['n2'] = n2max-n2min
    dictNew['jrange']=[x for x in range(0,dictNew['n2'])]
    dictNew['x2'] = dict['x2'][n2min:n2max]
    dictNew['dx2'] = dict['dx2'][n2min:n2max]
    dictNew['x2r'] = dict['x2r'][n2min:n2max]

    #set new x3 direction
    dictNew['n3_tot'] = 1
    dictNew['n3'] = 1
    dictNew['krange']=[0]
    dictNew['x3'] = dict['x3']
    dictNew['dx3'] = dict['dx3']
    dictNew['x3r'] = dict['x3r']

    for var in dict['vars']:
        dictNew[var]= dict[var][n1min:n1max,n2min:n2max]
    
    dataNew = copy.copy(data)
    dataNew.__dict__ = dictNew
    return dataNew
    
