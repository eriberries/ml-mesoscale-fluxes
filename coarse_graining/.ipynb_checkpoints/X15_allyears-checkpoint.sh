module load cdo

# Same steps as the X=5 version (refer to the file X5 for detailed comments)

runname="f.e22.F2000.NATL.branch"

indir= # source directory of the simulated outputs
outdir=../data/tempdata # store the processed files
tempdir=../data/tempdata # temporary location of temporary files
findir=../data/coarse # the final directory for the merged files

counter=1
mon=1
year=5

echo "START: Data selection and Flux computation"

for i in "$indir""$runname".cam.h3.* ; do
    echo $counter
    echo $year
    echo $mon
    # condition if there was an interruption
    p_start=1 # change this if you want to start from another file in case of interruption 
    if [ "$counter" -gt 1 ] ; then
        cdo select,name='U','V','OMEGA','Q','T','PS','CAPE','PRECC','PRECL' -seltimestep,1/60  "$i" "$tempdir"temp.nc
        ncap2 -s 'OMEGAU=OMEGA*U; OMEGAV=OMEGA*V; OMEGAT=OMEGA*T; OMEGAQ=OMEGA*Q' "$tempdir"temp.nc "$tempdir"temp2.nc
        rm "$tempdir"temp.nc
        ncremap -m "$indir"NATL.ne30x8_TO_f09-cnsrv.nc "$tempdir"temp2.nc "$tempdir"temp3.nc
        rm "$tempdir"temp2.nc
        cdo sellonlatbox,270,330,25,55 "$tempdir"temp3.nc "$tempdir"temp4.nc
        rm "$tempdir"temp3.nc
        ncap2 -s 'OMEGAU_Flux=OMEGAU-OMEGA*U; OMEGAV_Flux=OMEGAV-OMEGA*V; OMEGAT_Flux=OMEGAT-OMEGA*T; OMEGAQ_Flux=OMEGAQ-OMEGA*Q' "$tempdir"temp4.nc "$outdir"X15."$year"_mon_"$mon".f09regridded.nc
        rm "$tempdir"temp4.nc
    fi
    
    if ((counter % 12 == 0)); then  
        year=$((year + 1))
    fi
    counter=$((counter + 1))
    mon=$(((counter-1)%12 + 1))
done

echo "FINAL: merge the files to one dataset including all selected timesteps"
cdo mergetime "$outdir"X15.*_mon_*.f09regridded.nc "$findir"X15.allyears.f09regridded.nc


counter=1
mon=1  
year=5

echo "START: Sea level pressure data selection"
for i in "$indir""$runname".cam.h3.* ; do
    echo $counter
    echo $year
    echo $mon
    cdo select,name='PSL' -seltimestep,1/20  "$i" "$tempdir"temp.nc
    ncremap -m /net/krypton/climdyn/rjnglin/grids/var-res/ne0np4.NATL.ne30x8/NATL.ne30x8_TO_f09-cnsrv.nc "$tempdir"temp.nc "$tempdir"temp2.nc
    rm "$tempdir"temp.nc
    cdo sellonlatbox,270,330,25,55 "$tempdir"temp2.nc "$outdir"X15."$year"_mon_"$mon".PSL.f09regridded.nc
    rm "$tempdir"temp2.nc

    if ((counter % 12 == 0)); then  
        year=$((year + 1))
    fi
    counter=$((counter + 1))
    mon=$(((counter-1)%12 + 1))
done


echo "FINAL: merge the files to one dataset including all selected timesteps"
cdo mergetime "$outdir"X15.*_mon_*.PSL.f09regridded.nc "$findir"X15.allyears.PSL.f09regridded.nc

