#!/bin/bash
#SBATCH --job-name=piv         
#SBATCH --partition=cpu               # Take a node from the 'cpu' partition
#SBATCH --export=ALL                  # Export your environment to the compute node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20            # Ask for 10 CPU cores
#SBATCH --mem=50G                    # Memory request; MB assumed if unit not specified
#SBATCH --time=20:00:00               # Time limit hrs:min:sec
##SBATCH --output=%x-%j.log            # Standard output and error log
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


echo "computation start"
srun /scratch2/jsalle/.conda/envs/p11/bin/python3 robustness_check_deterministic.py 
echo "computation end" 
