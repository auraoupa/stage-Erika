cd /scratch/cnt0024/hmg2840/albert7a/eNATL60/eNATL60-BLBT02-S/1h/ACO
for file in $(ls eNATL60ACO-BLBT02_y2009m0?d??_sostrainoverf10m_filt2T.nc); do fileo=$(echo $file | sed 's/filt2T/filt2T_unlimited/g');  ncks --mk_rec_dmn time_counter $file -o $fileo; done
ncrcat eNATL60ACO-BLBT02_y2009m0?d??_sostrainoverf10m_filt2T_unlimited.nc eNATL60ACO-BLBT02_y2009m07-09_sostrainoverf10m_filt2T.nc

cd /scratch/cnt0024/hmg2840/albert7a/eNATL60/eNATL60-BLB002-S/1h/ACO
for file in $(ls eNATL60ACO-BLB002_y2009m0?d??_sostrainoverf10m_filt2T.nc); do fileo=$(echo $file | sed 's/filt2T/filt2T_unlimited/g');  ncks --mk_rec_dmn time_counter $file -o $fileo; done
ncrcat eNATL60ACO-BLB002_y2009m0?d??_sostrainoverf10m_filt2T_unlimited.nc eNATL60ACO-BLB002_y2009m07-09_sostrainoverf10m_filt2T.nc

for file in $(ls eNATL60ACO-BLB002_y2009m0?d??_socurloverf10m_filt2T.nc); do fileo=$(echo $file | sed 's/filt2T/filt2T_unlimited/g');  ncks --mk_rec_dmn time_counter $file -o $fileo; done
ncrcat eNATL60ACO-BLB002_y2009m0?d??_socurloverf10m_filt2T_unlimited.nc eNATL60ACO-BLB002_y2009m07-09_socurloverf10m_filt2T.nc
