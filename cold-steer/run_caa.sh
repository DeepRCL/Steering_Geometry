tasks=(hallucination refusal sycophancy survival-instinct myopic-reward coordinate-other-ais corrigible-neutral-HHH)
# tasks=(hallucination power-seeking wealth-seeking jailbreak)
# tasks=(sycophancy) # refusal sycophancy)
epsilons=(1e-6)
etas=(0.1 0.5 1 2)
# num_train_samples_list=(500 1000) #20 70 
num_train_samples_list=(50) #20 70 
maskings=(all) #(all last)
seeds=(42) #(123 93)


DATASET=caa # bipo_infer #
EVALUATION=infer
LLM=llama7b
# qwen7b_chat
layers=(5 10 15 20 30)
TEMPLATE=input_response # chat question_answer
GEN_MASKING=prompt
trainings=(dpo sft) # sft

for task in "${tasks[@]}"; do
    for training in "${trainings[@]}"; do
        python test.py \
            dataset=${DATASET}_${training}_${EVALUATION} \
            steerer=base \
            llm=${LLM} \
            dataset.task=${task} \
            steering_params.num_samples=1 \
            dataset.tokenizer_template=${TEMPLATE};
        
        for layer in "${layers[@]}"; do
            for eta in "${etas[@]}"; do
                for num_train_samples in "${num_train_samples_list[@]}"; do
                    for masking in "${maskings[@]}"; do
                        for seed in "${seeds[@]}"; do
                            # FD
                            python test.py \
                                dataset=${DATASET}_${training}_${EVALUATION} \
                                llm=${LLM} \
                                steerer=cold_fd \
                                dataset.task=${task} \
                                steerer.eta=${eta} \
                                steering_params.layers=[${layer}] \
                                steering_params.num_samples=${num_train_samples} \
                                steering_params.masking=${masking} \
                                +steering_params.gen_masking=${GEN_MASKING} \
                                seed=${seed} \
                                dataset.tokenizer_template=${TEMPLATE};

                            # Contrastive
                            if [[ "$training" == "dpo" ]]; then
                                python test.py \
                                    dataset=${DATASET}_${training}_${EVALUATION} \
                                    llm=${LLM} \
                                    steerer=contrastive \
                                    dataset.task=${task} \
                                    steerer.eta=${eta} \
                                    steering_params.layers=[${layer}] \
                                    steering_params.num_samples=${num_train_samples} \
                                    steering_params.masking=${masking} \
                                    +steering_params.gen_masking=${GEN_MASKING} \
                                    seed=${seed} \
                                    dataset.tokenizer_template=${TEMPLATE};
                            fi

                            # Reft vector
                            python test.py \
                                dataset=${DATASET}_${training}_${EVALUATION} \
                                llm=${LLM} \
                                steerer=reft \
                                steerer.intervention_type=vector \
                                dataset.task=${task} \
                                steering_params.layers=[${layer}] \
                                steering_params.num_samples=${num_train_samples} \
                                steering_params.masking=${masking} \
                                +steering_params.gen_masking=${GEN_MASKING} \
                                seed=${seed} \
                                dataset.tokenizer_template=${TEMPLATE};
                            
                            # # # Reft mlp
                            python test.py \
                                dataset=${DATASET}_${training}_${EVALUATION} \
                                llm=${LLM} \
                                steerer=reft \
                                steerer.intervention_type=mlp \
                                dataset.task=${task} \
                                steering_params.layers=[${layer}] \
                                steering_params.num_samples=${num_train_samples} \
                                steering_params.masking=${masking} \
                                +steering_params.gen_masking=${GEN_MASKING} \
                                seed=${seed} \
                                dataset.tokenizer_template=${TEMPLATE};

                            # # Kernel constant
                            python test.py \
                                dataset=${DATASET}_${training}_${EVALUATION} \
                                llm=${LLM} \
                                steerer=cold_kernel \
                                steerer.kernel=constant \
                                dataset.task=${task} \
                                steerer.eta=${eta} \
                                steering_params.layers=[${layer}] \
                                steering_params.num_samples=${num_train_samples} \
                                steering_params.masking=${masking} \
                                +steering_params.gen_masking=${GEN_MASKING} \
                                seed=${seed} \
                                dataset.tokenizer_template=${TEMPLATE};
                            
                            # # Kernel unit
                            python test.py \
                                dataset=${DATASET}_${training}_${EVALUATION} \
                                llm=${LLM} \
                                steerer=cold_kernel \
                                steerer.kernel=unit \
                                dataset.task=${task} \
                                steerer.eta=${eta} \
                                steering_params.layers=[${layer}] \
                                steering_params.num_samples=${num_train_samples} \
                                steering_params.masking=${masking} \
                                +steering_params.gen_masking=${GEN_MASKING} \
                                seed=${seed} \
                                dataset.tokenizer_template=${TEMPLATE};
                            
                            # # Kernel random_proj
                            python test.py \
                                dataset=${DATASET}_${training}_${EVALUATION} \
                                llm=${LLM} \
                                steerer=cold_kernel \
                                steerer.kernel=random_proj \
                                dataset.task=${task} \
                                steerer.eta=${eta} \
                                steering_params.layers=[${layer}] \
                                steering_params.num_samples=${num_train_samples} \
                                steering_params.masking=${masking} \
                                +steering_params.gen_masking=${GEN_MASKING} \
                                seed=${seed} \
                                dataset.tokenizer_template=${TEMPLATE};
                            
                        done
                    done
                done
            done
        done
    done
done


tasks=(hallucination refusal sycophancy survival-instinct myopic-reward coordinate-other-ais corrigible-neutral-HHH)
# tasks=(hallucination power-seeking wealth-seeking jailbreak)
# tasks=(sycophancy) # refusal sycophancy)
epsilons=(1e-6)
etas=(0.1 0.5 1 2)
# num_train_samples_list=(500 1000) #20 70 
num_train_samples_list=(50) #20 70 
maskings=(all) #(all last)
seeds=(42) #(123 93)


DATASET=caa # bipo_infer #
EVALUATION=infer
LLM=llama7b_chat
# qwen7b_chat
layers=(5 10 15 20 30)
TEMPLATE=chat # chat question_answer
GEN_MASKING=prompt
trainings=(dpo sft) # sft

for task in "${tasks[@]}"; do
    for training in "${trainings[@]}"; do
        python test.py \
            dataset=${DATASET}_${training}_${EVALUATION} \
            steerer=base \
            llm=${LLM} \
            dataset.task=${task} \
            steering_params.num_samples=1 \
            dataset.tokenizer_template=${TEMPLATE};
        
        for layer in "${layers[@]}"; do
            for eta in "${etas[@]}"; do
                for num_train_samples in "${num_train_samples_list[@]}"; do
                    for masking in "${maskings[@]}"; do
                        for seed in "${seeds[@]}"; do
                            # FD
                            python test.py \
                                dataset=${DATASET}_${training}_${EVALUATION} \
                                llm=${LLM} \
                                steerer=cold_fd \
                                dataset.task=${task} \
                                steerer.eta=${eta} \
                                steering_params.layers=[${layer}] \
                                steering_params.num_samples=${num_train_samples} \
                                steering_params.masking=${masking} \
                                +steering_params.gen_masking=${GEN_MASKING} \
                                seed=${seed} \
                                dataset.tokenizer_template=${TEMPLATE};

                            # Contrastive
                            if [[ "$training" == "dpo" ]]; then
                                python test.py \
                                    dataset=${DATASET}_${training}_${EVALUATION} \
                                    llm=${LLM} \
                                    steerer=contrastive \
                                    dataset.task=${task} \
                                    steerer.eta=${eta} \
                                    steering_params.layers=[${layer}] \
                                    steering_params.num_samples=${num_train_samples} \
                                    steering_params.masking=${masking} \
                                    +steering_params.gen_masking=${GEN_MASKING} \
                                    seed=${seed} \
                                    dataset.tokenizer_template=${TEMPLATE};
                            fi

                            # Reft vector
                            python test.py \
                                dataset=${DATASET}_${training}_${EVALUATION} \
                                llm=${LLM} \
                                steerer=reft \
                                steerer.intervention_type=vector \
                                dataset.task=${task} \
                                steering_params.layers=[${layer}] \
                                steering_params.num_samples=${num_train_samples} \
                                steering_params.masking=${masking} \
                                +steering_params.gen_masking=${GEN_MASKING} \
                                seed=${seed} \
                                dataset.tokenizer_template=${TEMPLATE};
                            
                            # # # Reft mlp
                            python test.py \
                                dataset=${DATASET}_${training}_${EVALUATION} \
                                llm=${LLM} \
                                steerer=reft \
                                steerer.intervention_type=mlp \
                                dataset.task=${task} \
                                steering_params.layers=[${layer}] \
                                steering_params.num_samples=${num_train_samples} \
                                steering_params.masking=${masking} \
                                +steering_params.gen_masking=${GEN_MASKING} \
                                seed=${seed} \
                                dataset.tokenizer_template=${TEMPLATE};

                            # # Kernel constant
                            python test.py \
                                dataset=${DATASET}_${training}_${EVALUATION} \
                                llm=${LLM} \
                                steerer=cold_kernel \
                                steerer.kernel=constant \
                                dataset.task=${task} \
                                steerer.eta=${eta} \
                                steering_params.layers=[${layer}] \
                                steering_params.num_samples=${num_train_samples} \
                                steering_params.masking=${masking} \
                                +steering_params.gen_masking=${GEN_MASKING} \
                                seed=${seed} \
                                dataset.tokenizer_template=${TEMPLATE};
                            
                            # # Kernel unit
                            python test.py \
                                dataset=${DATASET}_${training}_${EVALUATION} \
                                llm=${LLM} \
                                steerer=cold_kernel \
                                steerer.kernel=unit \
                                dataset.task=${task} \
                                steerer.eta=${eta} \
                                steering_params.layers=[${layer}] \
                                steering_params.num_samples=${num_train_samples} \
                                steering_params.masking=${masking} \
                                +steering_params.gen_masking=${GEN_MASKING} \
                                seed=${seed} \
                                dataset.tokenizer_template=${TEMPLATE};
                            
                            # # Kernel random_proj
                            python test.py \
                                dataset=${DATASET}_${training}_${EVALUATION} \
                                llm=${LLM} \
                                steerer=cold_kernel \
                                steerer.kernel=random_proj \
                                dataset.task=${task} \
                                steerer.eta=${eta} \
                                steering_params.layers=[${layer}] \
                                steering_params.num_samples=${num_train_samples} \
                                steering_params.masking=${masking} \
                                +steering_params.gen_masking=${GEN_MASKING} \
                                seed=${seed} \
                                dataset.tokenizer_template=${TEMPLATE};
                            
                        done
                    done
                done
            done
        done
    done
done

