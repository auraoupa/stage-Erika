#!/usr/bin/env python


from dask_jobqueue import SLURMCluster 
from dask.distributed import Client 
  
cluster = SLURMCluster(cores=28,name='make_profiles',walltime='00:30:00',job_extra=['--constraint=HSW24','--exclusive','--nodes=1'],memory='120GB',interface='ib0') 
cluster.scale(196)
cluster

from dask.distributed import Client
client = Client(cluster)
client



import time
nb_workers = 0
while True:
    nb_workers = len(client.scheduler_info()["workers"])
    if nb_workers >= 2:
        break
    time.sleep(1)
print(nb_workers)




## Path for modules

import sys

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






# Opening u & v grid files
# With tide

dirvar00='/scratch/cnt0024/hmg2840/albert7a/eNATL60/eNATL60-BLB002-S/1h/ACO/'
ufile00=dirvar00+'eNATL60ACO-BLB002_y2010m*.1h_vozocrtx10m.nc' # JFM
vfile00=dirvar00+'eNATL60ACO-BLB002_y2010m*.1h_vomecrty10m.nc' # JFM
dsu00=xr.open_mfdataset(ufile00,combine='by_coords',parallel=True,chunks={'time_counter':1008,'x':100,'y':1000})
dsv00=xr.open_mfdataset(vfile00,combine='by_coords',parallel=True,chunks={'time_counter':1008,'x':100,'y':1000})
u00=dsu00.vozocrtx
v00=dsv00.vomecrty 
lon=dsu00.nav_lon
lat=dsu00.nav_lat
time_counter=dsu00.time_counter

T=2*np.pi/(1E-4) # Coriolis period

wu00 = u00.window
wu00.set(n=48,dim='time_counter', cutoff=2*T)
u00_filt = wu00.convolve()

wv00 = v00.window
wv00.set(n=48,dim='time_counter', cutoff=2*T)
v00_filt = wv00.convolve()

### Compute curl and strain with python

curl00_filt   = curl(u00_filt,v00_filt,e1v,e2u,ff)
strain00_filt = strain(u00_filt,v00_filt,e1u,e1v,e2u,e2v,ff)

date_list=[]
for month in np.arange(1,4):
    if month == 1:
        dayf=31
    if month == 3:
        dayf=31
    if month == 2:
        dayf=28
    for day in np.arange(1,dayf+1):
        if day < 10:
            date='y2010m0'+str(month)+'d0'+str(day)
        else:
            date='y2010m0'+str(month)+'d'+str(day)
        date_list.append(date)




#for t in np.arange(len(date_list)):
#	print('writing file eNATL60ACO-BLB002_'+date_list[t]+'_sostrainoverf10m_filt2T.nc')
#	strain00_filt_1day = strain00_filt.squeeze()[24*t:24*(t+1)]
#	strain00_filt_da=xr.DataArray(strain00_filt_1day,dims=['time_counter','y','x'],name="strain_Tide_Filt")
#	strain00_filt_da.attrs['Name']='eNATL60ACO-BLB002_'+date_list[t]+'_sostrainoverf10m_filt2T.nc'
#	strain00_filt_da.to_dataset().to_netcdf(path='/scratch/cnt0024/hmg2840/albert7a/eNATL60/eNATL60-BLB002-S/1h/ACO/eNATL60ACO-BLB002_'+date_list[t]+'_sostrainoverf10m_filt2T.nc',mode='w',engine='scipy')

for t in np.arange(len(date_list)):
	print('writing file eNATL60ACO-BLB002_'+date_list[t]+'_socurloverf10m_filt2T.nc')
	curl00_filt_1day = curl00_filt.squeeze()[24*t:24*(t+1)]
	curl00_filt_da=xr.DataArray(curl00_filt_1day,dims=['time_counter','y','x'],name="curl_noTide_Filt")
	curl00_filt_da.attrs['Name']='eNATL60ACO-BLB002_'+date_list[t]+'_socurloverf10m_filt2T.nc'
	curl00_filt_da.to_dataset().to_netcdf(path='/scratch/cnt0024/hmg2840/albert7a/eNATL60/eNATL60-BLB002-S/1h/ACO/eNATL60ACO-BLB002_'+date_list[t]+'_socurloverf10m_filt2T.nc',mode='w',engine='scipy')




