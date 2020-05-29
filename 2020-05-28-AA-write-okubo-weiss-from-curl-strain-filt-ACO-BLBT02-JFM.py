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

import numpy.ma as ma

from datetime import date, datetime

import pandas as pd




# Opening grid files

dirgrid='/scratch/cnt0024/hmg2840/albert7a/eNATL60/eNATL60-I/'
gridfile=dirgrid+'mesh_hgr_eNATL60ACO_3.6.nc'


# Opening u & v grid files

dirvar00='/scratch/cnt0024/hmg2840/albert7a/eNATL60/eNATL60-BLBT02-S/1h/ACO/'
curlfile00=dirvar00+'eNATL60ACO-BLBT02_y2010m*_socurloverf10m_filt2T_unlimited.nc' # JFM
strainfile00=dirvar00+'eNATL60ACO-BLBT02_y2010m*_sostrainoverf10m_filt2T_unlimited.nc' # JFM
dscurl00=xr.open_mfdataset(curlfile00,combine='by_coords',parallel=True,chunks={'time_counter':1008,'x':100,'y':1000})
dsstrain00=xr.open_mfdataset(strainfile00,combine='by_coords',parallel=True,chunks={'time_counter':1008,'x':100,'y':1000})
curl00=dscurl00.curl_Tide_Filt
strain00=dsstrain00.strain_Tide_Filt
time_counter=dscurl00.time_counter

ow00=strain00**2 - curl00**2

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




for t in np.arange(len(date_list)):
	print('writing file eNATL60ACO-BLBT02_'+date_list[t]+'_ow10m_filt2T.nc')
	ow00_1day = ow00.squeeze()[24*t:24*(t+1)]
	ow00_da=xr.DataArray(ow00_1day,dims=['time_counter','y','x'],name="ow00_noTide_Filt")
	ow00_da.attrs['Name']='eNATL60ACO-BLBT02_'+date_list[t]+'_ow10m_filt2T.nc'
	ow00_da.to_dataset().to_netcdf(path='/scratch/cnt0024/hmg2840/albert7a/eNATL60/eNATL60-BLBT02-S/1h/ACO/eNATL60ACO-BLBT02_'+date_list[t]+'_ow10m_filt2T.nc',mode='w')




