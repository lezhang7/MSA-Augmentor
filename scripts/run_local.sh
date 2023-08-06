#!/bin/bash

#---- from alphafold2 run_docker.py -----#
# The following flags allow us to make predictions on proteins that
# would typically be too long to fit into GPU memory.
  export TF_FORCE_UNIFIED_MEMORY=1
  export XLA_PYTHON_CLIENT_MEM_FRACTION=4.0
# export XLA_PYTHON_CLIENT_PREALLOCATE=false


#---------------------------------------------------------#
##### ===== All functions are defined here ====== #########
#---------------------------------------------------------#

# ----- usage ------ #
usage()
{
	echo "alphafold2_sheng run_local v1.01 [Aug-11-2021] "
	echo "    Local run for alphafold2_sheng. "
	echo ""
	echo "USAGE:  ./run_local.sh <-i fasta | a3m> [-d data_root] [-a decoys] [-o out_root] "
	echo "                       [-m model_str] [-P seq_or_msa] [-T template_date] [-A run_amber] "
	echo "                       [-M a3m_type] [-g gpu_device] [-H home] [-E python_env] "
	echo "Options:"
	echo ""
	echo "***** required arguments *****"
	echo "-i fasta          : Query protein sequence in FASTA format. "
	echo "(or)"
	echo "-i a3m            : Query protein Multiple Sequence Alignments in a3m format. "
	echo ""
	echo "-d data_root      : database root for both MSA and PDB. "
	echo "                    [default = '/ssdcache/wangsheng/databases/' ] "
	echo ""
	echo "-a decoys         : The input decoys files. [default is null] "
	echo "                    [e.g., 'example/1pazA_1.pdb' ]"
	echo ""
	echo "-o out_root       : Output directory. [default = './\${input_name}_AF2'] "
	echo ""
	echo "***** optional arguments *****"
	echo "-m model_str      : output model number strings. "
	echo "                    [default = 'model_1,model_2,model_3,model_4,model_5' "
	echo ""
	echo "-P seq_or_msa     : pure_sequence or MSA mode. [default = 1 for MSA] "
	echo ""
	echo "-T template_date  : max_template_date. [default = 2021-05-14] "
	echo "                    [set -1 to disable templates]"
	echo ""
	echo "-A run_amber      : run amber or not to relax the model. [default = 1] "
	echo ""
	echo "-M a3m_type       : run a specific A3M type, and early break. [default = 'all'] "
	echo "                    [note]: type 'uniref90', 'mgnify', or 'bfd' to run and early break. "
	echo ""
	echo "-g gpu_device     : gpu_device. [default = '0' ] "
	echo ""
	echo "***** home relevant **********"
	echo "-H home           : home directory of alphafold2_sheng."
	echo "                    [default = `dirname $0`] "
	echo ""
	echo "-E python_env     : python environment of alphafold2_sheng. "
	echo "                    [default = '~/miniconda/envs/af2' ]"
	echo ""
	exit 1
}

#-------------------------------------------------------------#
##### ===== get pwd and check HomeDirectory ====== #########
#-------------------------------------------------------------#

#------ current directory ------#
curdir="$(pwd)"

#-------- check usage -------#
if [ $# -lt 1 ];
then
	usage
fi

#---------------------------------------------------------#
##### ===== All arguments are defined here ====== #########
#---------------------------------------------------------#


# ----- get arguments ----- #
#-> required arguments
input=""
input_fasta=""
input_a3m=""
A3M_or_NOT=0
decoys=""
data_root="/ssdcache/wangsheng/databases"
out_root=""
#-> optional arguments
model_str="model_1,model_2,model_3,model_4,model_5"
seq_or_msa="1"     #-> 1 for MSA and 0 for pure_seq
template_date="2021-05-14"  #-> type 1900-01-01 to cancel out ALL templates
run_amber="1"      #-> 1 for run amber 0 for not
a3m_type="all"
gpu_device="0"
#--| home relevant
home=`dirname $0`  #-> home directory
python_env="$HOME/miniconda/envs/af2"
#-> parse arguments
while getopts "i:a:d:o:m:P:T:A:M:g:H:E:" opt;
do
	case $opt in
	#-> required arguments
	i)
		input=$OPTARG
		;;
        a)
                decoys=$OPTARG
                ;;
	d)
		data_root=$OPTARG
		;;
	o)
		out_root=$OPTARG
		;;
	#-> optional arguments
	m)
		model_str=$OPTARG
		;;
	P)
		seq_or_msa=$OPTARG
		;;
	T)
		template_date=$OPTARG
		;;
	A)
		run_amber=$OPTARG
		;;
	M)
		a3m_type=$OPTARG
		;;
	g)
		gpu_device=$OPTARG
		;;
	#-> home relevant
	H)
		home=$OPTARG
		;;
	E)
		python_env=$OPTARG
		;;
	#-> help
	\?)
		echo "Invalid option: -$OPTARG" >&2
		exit 1
		;;
	:)
		echo "Option -$OPTARG requires an argument." >&2
		exit 1
		;;
	esac
done

#---------------------------------------------------------#
##### ===== Part 0: initial argument check ====== #########
#---------------------------------------------------------#
# ------ check home directory ---------- #
if [ ! -d "$home" ]
then
	echo "home directory $home not exist " >&2
	exit 1
fi
home=`readlink -f $home`
# ------ check python_env directory ---- #
echo $python_env
if [ ! -d "$python_env" ]
then
	echo "python_env directory $python_env not exist. Use '/usr' by default. " >&2
	python_env=/usr
fi
python_env=`readlink -f $python_env`
#----------- check input -----------#
if [ ! -s "$input" ]
then
	echo "input $input not found !!" >&2
	exit 1
fi
input=`readlink -f $input`
#-> get query_name
fulnam=`basename $input`
relnam=${fulnam%.*}
procnam=$relnam
# ------ judge fasta or tgt -------- #
filename=`basename $input`
extension=${filename##*.}
filename=${filename%.*}
if [ "$extension" == "a3m" ]
then
	input_a3m=$input
	A3M_or_NOT=1
else
	input_fasta=$input
	A3M_or_NOT=0
fi
# ------ check data root ----------------#
if [ ! -d "$data_root" ]
then
	echo "data_root $data_root not exist " >&2
	exit 1
fi
data_root=`readlink -f $data_root`
# ------ check output directory -------- #
if [ "$out_root" == "" ]
then
	out_root=$curdir/${relnam}_AF2
fi
mkdir -p $out_root
out_root=`readlink -f $out_root`

# ============ initialization ============== #
$home/util/Verify_FASTA $input $out_root/${relnam}.fasta


#-------- usage ------------#
# if [ $# -lt 4 ]
# then
#		echo "Usage:  ./run_local.sh <fasta_file> <out_root> <gpu_device> <python_env> "
#		echo "[note]: gpu_device starts from 0"
#		echo "        python_env shall be e.g., '~/miniconda/envs/af2' "
#		exit 1
# fi

#-------- required root ----#
out=$out_root
gpu=$gpu_device
file=$out/${relnam}.fasta
#-------- python env -------#
python=${python_env}/bin/python3
export PATH="${python_env}/bin:$PATH"


#-------- module load ------#
#module load cuda10.1
cuda=$(for i in `module avai | grep cuda`; do a=`echo $i | grep "^cuda"`; if [ "$a" != "" ]; then echo $a; break; fi ; done)
if [ "$cuda" != "" ]
then
	echo "cuda version is $cuda"
	module load cuda10.1 #$cuda
fi

#-------- check nvcc -------#
if ! command -v nvcc &> /dev/null
then
	echo "COMMAND nvcc could not be found !! try module load"
	exit 1
else
	nvcc -V
fi


#-------- pure_seq or MSA --#
if [ $seq_or_msa -ne 1 ]
then
	mkdir -p $out/msas
	$home/util/Verify_FASTA $file $out/msas/uniref90_hits.a3m
	cp $file $out/msas/mgnify_hits.a3m
	cp $file $out/msas/bfd_uniclust_hits.a3m
	touch $out/msas/pure_seq
else
	if [ -f $out/msas/pure_seq ]
	then
		rm -f $out/msas/uniref90_hits.a3m
		rm -f $out/msas/mgnify_hits.a3m
		rm -f $out/msas/bfd_uniclust_hits.a3m
		rm -f $out/msas/pure_seq
	fi
fi
#-------- run with A3M -----#
if [ $A3M_or_NOT -eq 1 ]
then
	mkdir -p $out/msas
	cp $input_a3m $out/msas/uniref90_hits.a3m
	cp $file $out/msas/mgnify_hits.a3m
	cp $file $out/msas/bfd_uniclust_hits.a3m
	touch $out/msas/input_a3m
else
	if [ -f $out/msas/input_a3m ]
	then
		rm -f $out/msas/uniref90_hits.a3m
		rm -f $out/msas/mgnify_hits.a3m
		rm -f $out/msas/bfd_uniclust_hits.a3m
		rm -f $out/msas/input_a3m
	fi
fi

#-------- run jobs ---------#
if [ $gpu -ge 0 ]
then
	export CUDA_VISIBLE_DEVICES=$gpu
fi
export AF2_DATA_DIR=$data_root
export AF2_HOME_DIR=$home
#-> run amber or not
amber_command=""
if [ "$run_amber" != "1" ]
then
	amber_command="--no_amber"
fi
#-> run template or not
run_template=""
if [ "$template_date" == "-1" ]
then
	run_template="--no_template"
        touch $out/msas/templates_hits.hhr
else
	run_template="--max_template_date=${template_date}"
fi
#----- run alphafold2 -------#
${python} ${home}/run_alphafold.py \
	--fasta_paths=$file \
	--decoys=$decoys \
	--model_names $model_str \
	--output_dir=$out \
	--a3m_type=$a3m_type \
	--preset=full_dbs \
	${run_template} ${amber_command}

#-------- post-process -----#
if [ "$a3m_type" == "all" ]
then
	$home/util/post_proc.sh $out/ $python_env
fi

# ========= exit 0 =========== #
exit 0

