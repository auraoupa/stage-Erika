concat_BLBT02=1
concat_BLB002=1

if [ $concat_BLBT02 -eq 1 ]; then
	cd /scratch/cnt0024/hmg2840/albert7a/eNATL60/eNATL60-BLBT02-S/1h/ACO
	for file in $(ls eNATL60ACO-BLBT02_y2010m0?d??_ow10m_filt2T.nc); do fileo=$(echo $file | sed 's/filt2T/filt2T_unlimited/g');  ncks -O --mk_rec_dmn time_counter $file -o $fileo; done
	ncrcat eNATL60ACO-BLBT02_y2010m0?d??_ow10m_filt2T_unlimited.nc eNATL60ACO-BLBT02_y2010m01-03_ow10m_filt2T.nc
fi

if [ $concat_BLB002 -eq 1 ]; then
	cd /scratch/cnt0024/hmg2840/albert7a/eNATL60/eNATL60-BLB002-S/1h/ACO
	for file in $(ls eNATL60ACO-BLB002_y2010m0?d??_ow10m_filt2T.nc); do fileo=$(echo $file | sed 's/filt2T/filt2T_unlimited/g');  ncks -O --mk_rec_dmn time_counter $file -o $fileo; done
	ncrcat eNATL60ACO-BLB002_y2010m0?d??_ow10m_filt2T_unlimited.nc eNATL60ACO-BLB002_y2010m01-03_ow10m_filt2T.nc

fi
