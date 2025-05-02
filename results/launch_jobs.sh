#!/usr/bin/env bash

# Default values for all jobs
DEFAULT_TEST_SIZE=10000
DEFAULT_EPOCHS=50
DEFAULT_RMTS_EPOCHS=50
DEFAULT_MODEL_TYPE="kuramoto"

# Set these to true or false to control flag inclusion
# If true, the flag will be INCLUDED in the command line
INCLUDE_USE_OMEGA=true
INCLUDE_DISABLE_BETWEEN=true
INCLUDE_SYMMETRIC_J=false
INCLUDE_ANIMATE=true

# Build suffix from flags for job naming
SUFFIX="K"
if [ "$INCLUDE_USE_OMEGA" = true ]; then SUFFIX="${SUFFIX}T"; else SUFFIX="${SUFFIX}F"; fi
if [ "$INCLUDE_DISABLE_BETWEEN" = true ]; then SUFFIX="${SUFFIX}T"; else SUFFIX="${SUFFIX}F"; fi
if [ "$INCLUDE_SYMMETRIC_J" = true ]; then SUFFIX="${SUFFIX}T"; else SUFFIX="${SUFFIX}F"; fi

# Function to submit a job
submit_job() {
    local GLYPHS=$1
    local SEED=$2
    local TRAIN_SIZE=$3
    local TIME=${4:-"1:00:00"}

    # Create a temporary script with a descriptive name
    TMP_SCRIPT="tmp_${SUFFIX}_g${GLYPHS}_s${SEED}.sh"
    
    # Start building the base command
    CMD="python kuramoto_main.py \
  --exp_name \"${SUFFIX}_g${GLYPHS}_s${SEED}\" \
  --seed ${SEED} \
  --n_train_glyphs ${GLYPHS} \
  --n_train ${TRAIN_SIZE} \
  --n_test ${DEFAULT_TEST_SIZE} \
  --epochs ${DEFAULT_EPOCHS} \
  --rmts_epochs ${DEFAULT_RMTS_EPOCHS} \
  --model_type \"${DEFAULT_MODEL_TYPE}\""

    # Add boolean flags only if they should be included
    if [ "$INCLUDE_USE_OMEGA" = true ]; then
        CMD="$CMD --use_omega"
    fi
    
    if [ "$INCLUDE_DISABLE_BETWEEN" = true ]; then
        CMD="$CMD --disable_between"
    fi
    
    if [ "$INCLUDE_SYMMETRIC_J" = true ]; then
        CMD="$CMD --symmetric_j"
    fi

    if [ "$INCLUDE_ANIMATE" = true ]; then
        CMD="$CMD --animate"
    fi
    
    # Create the script
    cat > $TMP_SCRIPT << EOF
#!/usr/bin/env bash
#SBATCH --job-name=${SUFFIX}_g${GLYPHS}_s${SEED}
#SBATCH --partition=scavenge_gpu
#SBATCH --time=${TIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --output=./slogs/%x_%A_%a.out
#SBATCH --error=./slogs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --requeue

module load miniconda
source activate osc

# For debugging - print command
echo "Running command: $CMD"

$CMD
EOF
    
    # Make it executable
    chmod +x $TMP_SCRIPT
    
    # Submit the job
    sbatch $TMP_SCRIPT
    
    # Show what we've done
    echo "Submitted job: ${SUFFIX}_g${GLYPHS}_s${SEED}"
    echo "  --use_omega: $([ "$INCLUDE_USE_OMEGA" = true ] && echo "included (true)" || echo "excluded (false)")"
    echo "  --disable_between: $([ "$INCLUDE_DISABLE_BETWEEN" = true ] && echo "included (true)" || echo "excluded (false)")" 
    echo "  --symmetric_j: $([ "$INCLUDE_SYMMETRIC_J" = true ] && echo "included (true)" || echo "excluded (false)")"
}

# Submit all jobs
# g15 jobs
submit_job 15 123 2000
submit_job 15 321 2000
submit_job 15 492 2000
submit_job 15 42 2000
submit_job 15 319 2000

# g50 jobs
submit_job 50 123 2000
submit_job 50 321 2000
submit_job 50 492 2000
submit_job 50 42 2000
submit_job 50 319 2000

# g95 jobs
submit_job 95 123 4000 "2:00:00"
submit_job 95 321 4000 "2:00:00"
submit_job 95 492 4000 "2:00:00"
submit_job 95 42 4000 "2:00:00"
submit_job 95 319 4000 "2:00:00"