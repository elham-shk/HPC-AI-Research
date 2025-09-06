#!/bin/bash

# Define total number of jobs and chunk size for 30 samples
total_jobs=24  # 30 samples divided into 3 chunks
chunk_size=100   

for (( i=0; i<$total_jobs; i++ )); do
    start=$(( i * chunk_size ))
    end=$(( start + chunk_size ))
    job_script="shapley_chunk_${start}_${end}.pbs"

    cat <<EOF > $job_script
#!/bin/bash
#PBS -N shap_chunk_${start}_${end}
#PBS -A WYOM0219
#PBS -q main
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=3:mem=20GB
#PBS -j oe

cd \$PBS_O_WORKDIR
module load conda/latest
conda activate my_environment

python run_shapley_chunk.py ${start} ${end}
EOF

    qsub $job_script
done
