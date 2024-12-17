#!/bin/bash
exit 0
# Number of GPUs
NUM_GPUS=5

# Maximum number of processes per GPU
PROCESSES_PER_GPU=1

# Lock file to manage the queue
LOCK_FILE=/tmp/gpu_lock_file

# Directory to store process status
STATUS_DIR=/tmp/gpu_status

# Create the status directory
mkdir -p $STATUS_DIR

# Initialize the GPU status files
for ((i=0; i<$NUM_GPUS; i++)); do
  echo 0 > $STATUS_DIR/gpu_$i
done

# Function to run your ML experiment
run_experiment() {
  DATA=$1
  ALGO=$2
  SEED=$3
  GPU=$4
  COUNT=$5
  N_BASIS=$6

  # make a log directory
  LOGDIR="text_logs/basis/$DATA/$ALGO/$SEED"
  mkdir -p $LOGDIR
  LOGFILE="$LOGDIR/log.txt"

  echo "Running experiment #$COUNT/$total_count: Dataset=$DATA, Alg=$ALGO, N_basis=$N_BASIS, Seed=$SEED on GPU $GPU"

  # TODO: Replace the following line with your actual experiment command
  # TODO: Ensure your command uses the specified GPU
  python test.py --alg $ALGO --dataset $DATA --seed $SEED --device $GPU --epochs $EPOCHS --log_dir logs/basis --n_basis $N_BASIS > $LOGFILE 2>&1

  # get the exit code, print a warning if bad
  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    echo "WARNING: Experiment #$COUNT failed with exit code $EXIT_CODE"
  fi

  # After completion, decrement the GPU process count
  flock $LOCK_FILE bash -c "count=\$(< $STATUS_DIR/gpu_$GPU); echo \$((count - 1)) > $STATUS_DIR/gpu_$GPU"

  # let it deallocate memory
  sleep 1
}

export -f run_experiment
export STATUS_DIR
export LOCK_FILE

# Function to manage the queue and distribute jobs
manage_queue() {
  while read -r JOB; do
    DATA=$(echo $JOB | cut -d ' ' -f 1)
    ALGO=$(echo $JOB | cut -d ' ' -f 2)
    SEED=$(echo $JOB | cut -d ' ' -f 3)
    COUNT=$(echo $JOB | cut -d ' ' -f 4)
    N_BASIS=$(echo $JOB | cut -d ' ' -f 5)

    # Loop to find an available GPU
    while true; do
      for ((i=0; i<$NUM_GPUS; i++)); do
        # Lock the file and check the GPU status
        if flock $LOCK_FILE bash -c "[ \$(< $STATUS_DIR/gpu_$i) -lt $PROCESSES_PER_GPU ]"; then
          # Update the GPU status and start the experiment
          flock $LOCK_FILE bash -c "count=\$(< $STATUS_DIR/gpu_$i); echo \$((count + 1)) > $STATUS_DIR/gpu_$i"
          run_experiment $DATA $ALGO $SEED $i $COUNT $N_BASIS &
          break 2
        fi
      done
      # Wait before retrying to find an available GPU
      sleep 2
    done
    # Wait before starting the next job
    sleep 2
  done
  wait
}

ALGS="LS IP"
DATASETS="Polynomial CIFAR 7Scenes Ant"
N_BASIS="1 2 3 5 10 20 40 60 80 100"
EPOCHS=50000
job_list=()
total_count=0

# Main experiments using L2 based loss
for dataset in $DATASETS; do
   for alg in $ALGS; do
     for n_basis in $N_BASIS; do
       for seed in {1..3}; do
         job_list+=("$dataset $alg $seed $total_count $n_basis")
         total_count=$((total_count + 1))
       done
     done
   done
done

# Convert job list to a format suitable for the manage_queue function
printf "%s\n" "${job_list[@]}" | manage_queue



