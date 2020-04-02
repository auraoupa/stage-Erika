#!/usr/bin/env python

## Path for modules

import sys
import datetime
import numpy as np
import xarray as xr

from netCDF4 import Dataset

sys.path.insert(0,"/scratch/cnt0024/hmg2840/albert7a/DEV/git/xscale")
import xscale
import xscale.spectral.fft as xfft

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import numpy.ma as ma

import matplotlib.cm as mplcm
import matplotlib.colors as colors
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import date, datetime
#from xhistogram.xarray import histogram

import pandas as pd

import seaborn as sns
sns.set(color_codes=True)





def curl(u,v,e1v,e2u,ff):
    '''
    This routine computes the relative vorticity from 2D fields of horizontal velocities and the spatial Coriolis parameter.
    '''
    #Computation of dy(u)
    fe2u=1/e2u
    fse2u=fe2u.squeeze()
    dyu=(u.shift(y=-1) - u)*fse2u
    #Computation of dx(v)
    fe1v=1/e1v
    fse1v=fe1v.squeeze()
    dxv=(v.shift(x=-1) - v)*fse1v
    #Projection on the grid T
    dxvt=0.5*(dxv.shift(y=-1)+dxv)
    dyut=0.5*(dyu.shift(x=-1)+dyu)
    #Computation of the vorticity divided by f
    fff=1/ff
    curl=(dxvt-dyut)*fff
    return curl




def strain(u,v,e1u,e1v,e2u,e2v,ff):
    '''
    This routine computes the relative vorticity from 2D fields of horizontal velocities and the spatial Coriolis parameter.
    '''
    #Computation of dy(u)
    fe2u=1/e2u
    fse2u=fe2u.squeeze()
    dyu=(u.shift(y=-1) - u)*fse2u
    #Computation of dx(v)
    fe1v=1/e1v
    fse1v=fe1v.squeeze()
    dxv=(v.shift(x=-1) - v)*fse1v
    #Computation of dy(v)
    fe2v=1/e2v
    fse2v=fe2v.squeeze()
    dyv=(v.shift(y=-1) - v)*fse2v
    #Computation of dx(u)
    fe1u=1/e1u
    fse1u=fe1u.squeeze()
    dxu=(u.shift(x=-1) - u)*fse1u
    #Projection on the grid T
    dxvt=0.5*(dxv.shift(y=-1)+dxv)
    dyut=0.5*(dyu.shift(x=-1)+dyu)
    dxut=0.5*(dxu.shift(x=-1)+dxu)
    dyvt=0.5*(dyv.shift(y=-1)+dyv)
    #Computation of the strain divided by f
    fff=1/ff
    strain=np.sqrt( (dyut+dxvt)*(dyut+dxvt) + (dxut-dyvt)*(dxut-dyvt) ) *fff
    return strain




## Dataset : grid files and one example of U and V for test


# Opening grid files

dirgrid='/scratch/cnt0024/hmg2840/albert7a/eNATL60/eNATL60-I/'
gridfile=dirgrid+'mesh_hgr_eNATL60ACO_3.6.nc'
dsgrid=xr.open_dataset(gridfile,chunks={'x':200,'y':200})

e1u=dsgrid.e1u
e1v=dsgrid.e1v
e2u=dsgrid.e2u
e2v=dsgrid.e2v
ff=dsgrid.ff






def test_write_1netcdf(chunkt,chunkx,chunky):
    # Opening u & v grid files
    # With tide

    dirvarT0='/scratch/cnt0024/hmg2840/albert7a/eNATL60/eNATL60-BLBT02-S/1h/ACO/'
    ufileT0=dirvarT0+'eNATL60ACO-BLBT02_y2009m*.1h_vozocrtx10m.nc' # JAS
    vfileT0=dirvarT0+'eNATL60ACO-BLBT02_y2009m*.1h_vomecrty10m.nc' # JAS
    dsuT0=xr.open_mfdataset(ufileT0,combine='by_coords',parallel=True,chunks={'time_counter':chunkt,'x':chunkx,'y':chunky})
    dsvT0=xr.open_mfdataset(vfileT0,combine='by_coords',parallel=True,chunks={'time_counter':chunkt,'x':chunkx,'y':chunky})
    uT0=dsuT0.vozocrtx
    vT0=dsvT0.vomecrty 
    lon=dsuT0.nav_lon
    lat=dsuT0.nav_lat


    T=2*np.pi/(1E-4) # Coriolis period

    wuT0 = uT0.window
    wuT0.set(n=48,dim='time_counter', cutoff=2*T)
    uT0_filt = wuT0.convolve()

    wvT0 = vT0.window
    wvT0.set(n=48,dim='time_counter', cutoff=2*T)
    vT0_filt = wvT0.convolve()

    ### Compute curl and strain with python

    curlT0_filt   = curl(uT0_filt,vT0_filt,e1v,e2u,ff)
    

    curlT0_filt_da=xr.DataArray(curlT0_filt.squeeze(),dims=['time_counter','y','x'],name="Curl_Tide_Filt")
    curlT0_filt_da.attrs['Name']='eNATL60ACO-BLBT02_y2009m07-09_socurloverf10m_filt2Tb.nc'
    curlT0_filt_da.to_dataset().to_netcdf(path='/scratch/cnt0024/hmg2840/albert7a/eNATL60/eNATL60-BLBT02-S/1h/ACO/eNATL60ACO-BLBT02_y2009m07-09_socurloverf10m_filt2Tb.nc',mode='w',engine='scipy')

#print (str(datetime.now()))
#test_write_1netcdf(24,1000,1000)
print (str(datetime.now()))


test_write_1netcdf(1008,100,1000)
print (str(datetime.now()))


