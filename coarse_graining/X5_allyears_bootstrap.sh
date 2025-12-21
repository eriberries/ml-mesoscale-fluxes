module load cdo

# This script selects the data for bootstrapping, which includes the remaining data for the time span of the test set.

runname="f.e22.F2000.NATL.branch"

indir= # source directory of the simulated outputs
outdir=../data/tempdata # store the processed files
tempdir=../data/tempdata # temporary location of temporary files
findir=../data/coarse # the final directory for the merged files


# rm "$outdir"X5.*_mon_*.f09regridded.val.nc

counter=1
mon=1 # does not correspond to the right month of the first dataset, but it helps to keep order
year=5


echo "START: Data selection and Flux computation"

for file in "$indir""$runname".cam.h3.*; do
    # Extract the 4-digit number after cam.h3. and before the dash
    num=$(echo "$file" | sed -n 's/.*\.cam\.h3\.\([0-9]\{4,\}\)-.*/\1/p')

    # Check if it's a valid number and >= 32, to get the time span 
    if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 32 ]; then
        echo "Processing: $file"
        
        # keep track of the file which is being processed
        echo $counter
        echo $year
        echo $mon
        nsteps=$(cdo -ntime "$file")
        echo number of time steps: $nsteps

        cdo select,name='U','V','OMEGA','Q','T','PS','CAPE','PRECC','PRECL' -seltimestep,21/$nsteps  "$file" "$tempdir"temp.nc
      
        # compute the fluxes
        ncap2 -s 'OMEGAU=OMEGA*U; OMEGAV=OMEGA*V; OMEGAT=OMEGA*T; OMEGAQ=OMEGA*Q' "$tempdir"temp.nc "$tempdir"temp2.nc
        rm "$tempdir"temp.nc
      
        # remap conservatively to the f09 grid (100km)
        ncremap -m "$indir"NATL.ne30x8_TO_f09-cnsrv.nc "$tempdir"temp2.nc "$tempdir"temp3.nc
        rm "$tempdir"temp2.nc
      
        # already select the lon-lat box to reduce the amount of data stored
        cdo sellonlatbox,270,330,25,55 "$tempdir"temp3.nc "$tempdir"temp4.nc
        rm "$tempdir"temp3.nc
      
        # compute also the subgrid-scale fluxes 
        ncap2 -s 'OMEGAU_Flux=OMEGAU-OMEGA*U; OMEGAV_Flux=OMEGAV-OMEGA*V; OMEGAT_Flux=OMEGAT-OMEGA*T; OMEGAQ_Flux=OMEGAQ-OMEGA*Q' "$tempdir"temp4.nc "$outdir"X5."$year"_mon_"$mon".f09regridded.val.nc
        rm "$tempdir"temp4.nc
      
        if ((counter % 12 == 0)); then  # every 12 steps move to the next year
            year=$((year + 1))
        fi
        counter=$((counter + 1))
        mon=$(((counter-1)%12 + 1))
    fi
done

echo "FINAL: merge the files to one dataset including all selected timesteps"
cdo mergetime "$outdir"X5.*_mon_*.f09regridded.val.nc "$findir"X5.allyears.f09regridded.bootstrap.nc



# Sea Level pressure for visualization, repeat the same steps as above leaving out the flux calculation
echo "START: Sea level pressure data selection"

counter=1
mon=1  
year=5



for file in "$indir""$runname".cam.h3.*; do
    # Extract the 4-digit number after cam.h3. and before the dash
    num=$(echo "$file" | sed -n 's/.*\.cam\.h3\.\([0-9]\{4,\}\)-.*/\1/p')

    # Check if it's a valid number and >= 32
    if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 32 ]; then
        echo "Processing: $file"
        
        echo $counter
        echo $year
        echo $mon
        cdo select,name='PSL' -seltimestep,21/40  "$file" "$tempdir"temp.nc
        ncremap -m /net/krypton/climdyn/rjnglin/grids/var-res/ne0np4.NATL.ne30x8/NATL.ne30x8_TO_f09-cnsrv.nc "$tempdir"temp.nc "$tempdir"temp2.nc
        rm "$tempdir"temp.nc
        cdo sellonlatbox,270,330,25,55 "$tempdir"temp2.nc "$outdir"X5."$year"_mon_"$mon".PSL.f09regridded.val.nc
        rm "$tempdir"temp2.nc
    
        if ((counter % 12 == 0)); then  
            year=$((year + 1))
        fi
        counter=$((counter + 1))
        mon=$(((counter-1)%12 + 1))
    fi
done


echo "FINAL: merge the files to one dataset including all selected timesteps"
cdo mergetime "$outdir"X5.*_mon_*.PSL.f09regridded.val.nc "$findir"X5.allyears.PSL.f09regridded.val.nc

