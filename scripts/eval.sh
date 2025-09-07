# sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-233-232:30000/v1 --mode cot --data-name 2wiki &&
# sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-233-232:30000/v1 --mode mcts --data-name 2wiki &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-233-232:30000/v1 --mode cot --data-name hotpotqa &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-233-232:30000/v1 --mode mcts --data-name hotpotqa &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-233-232:30000/v1 --mode cot --data-name musique &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-233-232:30000/v1 --mode mcts --data-name musique &&

echo "All jobs submitted."
