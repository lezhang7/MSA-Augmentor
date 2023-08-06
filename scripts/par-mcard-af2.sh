#!/bin/bash
#SBATCH -J df-gen
#SBATCH -N 1
#SBATCH -n 104
#SBATCH --gres=gpu:8
#SBATCH --mem-per-cpu=300000MB
#SBATCH --time=48:00:00


# af2 task launcher
# Usage: ./par-mcard-af2.sh $msa_dir $output_dir
# note: all dir params need to end with '/'
# each msa file relate to a result directory of the same name in output_dir
# customize: edit $degree and $tar_card to set cards for potential usage


###################### find spare file descriptor ############################
found=none
for fd in {3..200} ; do
    [[ ! -e /proc/$$/fd/${fd} ]] && found=${fd} && break
done
echo "First free fd is ${found}"  # find the first spare file descriptor


###################### prepared samaphore setup  #############################
_fifofile="$$.fifo"
mkfifo $_fifofile     # create a FIFO type file
link_fd="exec $found<> $_fifofile"   # combine the fd with FIFO
eval $link_fd
rm $_fifofile         # clear

tar_card=(0 1 2 3 4 5 6 7)  # set the card number for tasks
degree=4  # define tasks on basha single card

# set signals for each card
# in fact it put $degree each card number into the file
# for ((i=0;i<${degree};i++));do
#     for j in ${tar_card[*]};do
#         echo -n "$j"
#     done
# done >&"${found}"

mkdir -p $2
module load cuda10.1
######################   launch parallel tasks    ############################
for fp in "$1"*  # arg pass in as a3m directory
do
  filename=`basename "$fp"`
  fname=${filename%.*}
  #echo "$fname"
  read -u"${found}" -n1 card_order  # read a char into param $card_order
  {
    echo "current $card_order $fname" 
    # call af2 modeling script
    [ -f "$2$fname/ranked_0.pdb" ] && echo "$fname exists" || \
        run_local.sh \
            -i "$fp" \
            -o "$2""$fname""/" \
            -T -1 -g "$card_order" \
            -E /share/wangsheng/miniconda/envs/af2"
    sleep 3
    echo -n "$card_order">&"${found}"  # store the char $card_order back into fd
  } &  
done

wait # wait for all tasks to finish

unlink_fd="exec $found>&-" # close the pipe
eval $unlink_fd

