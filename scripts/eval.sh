sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-224-9:30000/v1 --mode cot --data-name 2wiki &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-224-9:30000/v1 --mode mcts --data-name 2wiki &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-224-9:30000/v1 --mode cot --data-name hotpotqa &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-224-9:30000/v1 --mode mcts --data-name hotpotqa &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-224-9:30000/v1 --mode cot --data-name musique &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-224-9:30000/v1 --mode mcts --data-name musique &&

sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-224-9:30000/v1 --mode cot --data-name simpleqa &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-224-9:30000/v1 --mode mcts --data-name simpleqa &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-224-9:30000/v1 --mode cot --data-name bamboogle &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-224-9:30000/v1 --mode mcts --data-name bamboogle &&

sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-224-9:30000/v1 --mode cot --data-name nq &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-224-9:30000/v1 --mode mcts --data-name nq &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-224-9:30000/v1 --mode cot --data-name triviaqa &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-224-9:30000/v1 --mode mcts --data-name triviaqa &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-224-9:30000/v1 --mode cot --data-name popqa &&
sbatch scripts/run_infer_slurm.sh --ext-model Hieuman/Extractor-Qwen3-4B-SFT-v1 --ext-url http://ip-10-4-224-9:30000/v1 --mode mcts --data-name popqa &&


echo "All jobs submitted."
