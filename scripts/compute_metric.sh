python evaluate.py mode=mcts data.name=2wiki to_eval_path=results/mcts/Generator_openai-qwen3-8B/Extractor_Hieuman-Extractor-Qwen3-4B-SFT-v1/Evaluator_openai-qwen3-8B/2wiki/
python evaluate.py mode=cot data.name=2wiki to_eval_path=results/cot/Generator_openai-qwen3-8B/Extractor_Hieuman-Extractor-Qwen3-4B-SFT-v1/Evaluator_openai-qwen3-8B/2wiki/

python evaluate.py mode=mcts data.name=bamboogle to_eval_path=results/cot/Generator_bedrock-us.anthropic.claude-3-7-sonnet-20250219-v1:0/Extractor_Hieuman-Extractor-Qwen3-4B-SFT-v1/Evaluator_bedrock-us.anthropic.claude-3-7-sonnet-20250219-v1:0/bamboogle/
python evaluate.py mode=cot data.name=bamboogle to_eval_path=results/mcts/Generator_Qwen-Qwen3-30B-A3B-Thinking-2507/Extractor_Hieuman-Extractor-Qwen3-4B-SFT-v1/Evaluator_Qwen-Qwen3-30B-A3B-Thinking-2507/bamboogle/

python evaluate.py mode=mcts data.name=hotpotqa to_eval_path=results/mcts/Generator_openai-qwen3-8B/Extractor_Hieuman-Extractor-Qwen3-4B-SFT-v1/Evaluator_openai-qwen3-8B/hotpotqa/
python evaluate.py mode=cot data.name=hotpotqa to_eval_path=results/cot/Generator_openai-qwen3-8B/Extractor_Hieuman-Extractor-Qwen3-4B-SFT-v1/Evaluator_openai-qwen3-8B/hotpotqa/

python evaluate.py mode=mcts data.name=musique to_eval_path=results/mcts/Generator_openai-qwen3-8B/Extractor_Hieuman-Extractor-Qwen3-4B-SFT-v1/Evaluator_openai-qwen3-8B/musique/
python evaluate.py mode=cot data.name=musique to_eval_path=results/cot/Generator_openai-qwen3-8B/Extractor_Hieuman-Extractor-Qwen3-4B-SFT-v1/Evaluator_openai-qwen3-8B/musique/

python evaluate.py mode=mcts data.name=nq to_eval_path=results/mcts/Generator_openai-qwen3-8B/Extractor_Hieuman-Extractor-Qwen3-4B-SFT-v1/Evaluator_openai-qwen3-8B/nq/
python evaluate.py mode=cot data.name=nq to_eval_path=results/cot/Generator_openai-qwen3-8B/Extractor_Hieuman-Extractor-Qwen3-4B-SFT-v1/Evaluator_openai-qwen3-8B/nq/

python evaluate.py mode=mcts data.name=popqa to_eval_path=results/mcts/Generator_openai-qwen3-8B/Extractor_Hieuman-Extractor-Qwen3-4B-SFT-v1/Evaluator_openai-qwen3-8B/popqa/
python evaluate.py mode=cot data.name=popqa to_eval_path=results/cot/Generator_openai-qwen3-8B/Extractor_Hieuman-Extractor-Qwen3-4B-SFT-v1/Evaluator_openai-qwen3-8B/popqa/

python evaluate.py mode=mcts data.name=simpleqa to_eval_path=results/mcts/Generator_openai-qwen3-8B/Extractor_Hieuman-Extractor-Qwen3-4B-SFT-v1/Evaluator_openai-qwen3-8B/simpleqa/
python evaluate.py mode=cot data.name=simpleqa to_eval_path=results/cot/Generator_openai-qwen3-8B/Extractor_Hieuman-Extractor-Qwen3-4B-SFT-v1/Evaluator_openai-qwen3-8B/simpleqa/

python evaluate.py mode=mcts data.name=triviaqa to_eval_path=results/mcts/Generator_openai-qwen3-8B/Extractor_Hieuman-Extractor-Qwen3-4B-SFT-v1/Evaluator_openai-qwen3-8B/triviaqa/
python evaluate.py mode=cot data.name=triviaqa to_eval_path=results/cot/Generator_openai-qwen3-8B/Extractor_Hieuman-Extractor-Qwen3-4B-SFT-v1/Evaluator_openai-qwen3-8B/triviaqa/
