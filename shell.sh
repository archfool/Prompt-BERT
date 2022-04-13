python evaluation_rzh.py \
    --model_name_or_path   D:\data\huggingface\bert-base-chinese \
    --pooler avg \
    --mode test \
    --mask_embedding_sentence \
    --mask_embedding_sentence_template "*cls*这个句子：\"*sent_0*\"的意思是：*mask*。*sep+*" \
    --mask_embedding_sentence_delta

#prompt mask: --mask_embedding_sentence
#prompt mask, template denoising: --mask_embedding_sentence_delta
#cls: --pooler cls
#底层avg: --pooler avg  --embedding_only
#顶层avg: --pooler avg
#首尾两层avg: --pooler avg_first_last
#顶部两层avg: --pooler avg_top2


python train_rzh.py \
    --model_name_or_path D:\data\huggingface\bert-base-chinese \
    --train_file D:\data\SimCSE\wiki1m_for_simcse.txt \
    --output_dir D:\data\SimCSE\output \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --metric_for_best_model stsb_spearman \
    --evaluation_strategy no \
    --eval_steps 1 \
    --overwrite_output_dir \
    --mask_embedding_sentence \
    --mask_embedding_sentence_template "*cls*_This_sentence_:_\"*sent_0*\"_means*mask*.*sep+*"

# D:\data\SimCSE\wiki1m_for_simcse.txt
# D:\data\SimCSE\nli_for_simcse.csv


python evaluation.py \
    --model_name_or_path   D:\data\huggingface\bert-base-uncased \
    --pooler avg \
    --mode test \
    --mask_embedding_sentence \
    --mask_embedding_sentence_template "*cls*_This_sentence_:_\"*sent_0*\"_means*mask*.*sep+*"

python train.py \
    --model_name_or_path D:\data\huggingface\bert-base-uncased \
    --train_file D:\data\SimCSE\nli_for_simcse.csv \
    --output_dir D:\data\SimCSE\output \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --metric_for_best_model stsb_spearman \
    --evaluation_strategy no \
    --eval_steps 1 \
    --overwrite_output_dir \
    --mask_embedding_sentence \
    --mask_embedding_sentence_template "*cls*_This_sentence_:_\"*sent_0*\"_means*mask*.*sep+*"


