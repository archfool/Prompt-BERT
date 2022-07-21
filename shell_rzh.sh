"*cls*这个句子：*sent_0*的意思是：*mask*。*sep+*"
"*cls*这个句子：\"*sent_0*\"的意思是：*mask*。*sep+*"


python evaluation_rzh.py \
    --ori_model_name_or_path   D:\data\huggingface\bert-base-chinese \
    --model_name_or_path   D:\data\AutoPromptBERT\delta_org-mlp_1234 \
    --pooler avg \
    --mode test \
    --mask_embedding_sentence \
    --mask_embedding_sentence_template "*cls*这个句子：*sent_0*的意思是：*mask*。*sep+*" \
    --mask_embedding_sentence_autoprompt \
    --mask_embedding_sentence_delta \
    --mask_embedding_sentence_org_mlp \
    ;

#autoprompt mask: --mask_embedding_sentence --mask_embedding_sentence_autoprompt
#prompt mask: --mask_embedding_sentence
#prompt mask, template denoising: --mask_embedding_sentence_delta
#cls: --pooler cls
#底层avg: --pooler avg  --embedding_only
#顶层avg: --pooler avg
#首尾两层avg: --pooler avg_first_last
#顶部两层avg: --pooler avg_top2
#模板校正: --mask_embedding_sentence_delta
#使用原生bert的cls.predictions.transform做输出层变换: --mask_embedding_sentence_org_mlp



#autoprompt有监督训练(使用cls作为句子表征)
nohup sh -c '\
for seed in 1234 4321 1248 1357 2468;
do
  for delta_flag in true false;
  do
    for org_mlp_flag in true false;
    do
      python train_rzh.py \
          --model_name_or_path D:\data\huggingface\bert-base-chinese \
          --train_file D:\data\chinese_corpus\sim_sents.csv \
          --output_dir D:\data\AutoPromptBERT \
          --per_device_train_batch_size 128 \
          --learning_rate 3e-5 \
          --max_seq_length 64 \
          --do_train \
          --num_train_epochs 1 \
          --evaluation_strategy no \
          --save_strategy steps \
          --save_steps 100 \
          --save_total_limit 10 \
          --overwrite_output_dir \
          --mask_embedding_sentence \
          --mask_embedding_sentence_template "*cls*这个句子：*sent_0*的意思是：*mask*。*sep+*" \
          --mask_embedding_sentence_autoprompt \
          --mask_embedding_sentence_delta ${delta_flag} \
          --mask_embedding_sentence_org_mlp ${org_mlp_flag} \
          --seed ${seed}
    done
  done
done
' >> /media/archfool/data/data/SemEval-2022/task9/log/runoob.log 2>&1 &

#中文文本相似度语料
#--train_file D:\data\chinese_corpus\sim_sents.csv

#    --train_file D:\data\chinese_corpus\sim_sents.csv \
#    --train_file D:\data\SimCSE\nli_for_simcse.csv \
#（拟废弃，代码不严谨）是否使用模板embedding校正
#    --mask_embedding_sentence_delta \
#是否使用原生bert的cls.predictions.transform做输出层处理
#    --mask_embedding_sentence_org_mlp \

#基础训练
python train_rzh.py \
    --model_name_or_path D:\data\huggingface\bert-base-chinese \
    --train_file D:\data\chinese_corpus\sim_sents.csv \
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


