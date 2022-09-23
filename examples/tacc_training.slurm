#!/bin/bash

#SBATCH -J xray     # Job name
#SBATCH -o job_outputs/myjob.o%j       # Name of stdout output file
#SBATCH -e job_outputs/myjob.e%j       # Name of stderr error file
#SBATCH -p normal        # Queue (partition) name
#SBATCH -N 1               # Total # of nodes 
#SBATCH -n 1               # Total # of mpi tasks
#SBATCH -t 48:00:00         # Run time (hh:mm:ss)
#SBATCH --mail-user=jonathan.wittmer@utexas.edu
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A DMS22002

# Other commands must follow all #SBATCH directives...

#module list
#pwd
#date
module load python3
export OMP_NUM_THREADS=128
# Launch MPI code... 

#ibrun python3 train_mangll.py ${json_options}       # Use ibrun instead of mpirun or mpiexec
echo here
cd xray_tomography
python3 xray_tomography.py
# ---------------------------------------------------
