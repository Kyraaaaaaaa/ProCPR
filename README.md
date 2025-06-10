# ProCPR

## Download Raw Gowalla dataset under dataset folder
The Gowalla dataset is downloaded from http://www.yongliu.org/datasets/ .
We extract the check-in records in two cross-city pairs, including Dallas-Austin and LosAngeles-SanFrancisco. 

File Gowalla_Dallas_Austin.csv and Gowalla_LosAngeles_SanFrancisco.csv contain check-ins with 7 columns, which are:
1. userid (anonymized)
2. placeid (ID of check-in POI)
3. datetime (UTC time)
4. lng (longitude of check-in POI)
5. lat (latitude of check-in POI)
6. spot_categ (category of check-in POI)
7. cross_city_mode ("A_B" stands for user from city A in city B)

## Data Preprocessing
### LosAngeles-SanFrancisco
* ```cd data_preprocessing```

* ```python cross_city_data_preprocessing.py --city_A=LosAngeles --city_B=SanFrancisco```

* ```python to_nextpoi_kqt_geo.py --city_A=LosAngeles --city_B=SanFrancisco```

* ```python generate_user_profile.py --city_A=LosAngeles --city_B=SanFrancisco```
* ```python create_sft_dataset_traj_total.py --city_A=LosAngeles --city_B=SanFrancisco```



### SanFrancisco-LosAngeles
* ```cd data_preprocessing```
* ```python cross_city_data_preprocessing.py --city_A=SanFrancisco --city_B=LosAngeles```
* ```python create_sft_dataset_traj_total.py --city_A=SanFrancisco --city_B=LosAngeles```
* ```python generare_user_profile.py --city_A=SanFrancisco --city_B=LosAngeles```
* ```python to_nextpoi_kqt_geo.py --city_A=SanFrancisco --city_B=LosAngeles```

## Train Model
```
torchrun --nproc_per_node=2 supervised-fine-tune-qlora.py  \
--model_name_or_path Llama-2-7b-longlora-32k-ft \
--bf16 True \
--output_dir dataset/Gowalla/SanFrancisco-LosAngeles/models/base_models \
--model_max_length 32768 \
--use_flash_attn True \
--data_path UPTDNet_dataset/Gowalla/SanFrancisco-LosAngeles/train_total.json \
--low_rank_training True \
--num_train_epochs 5 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--save_total_limit 5 \
--learning_rate 2e-4 \
--weight_decay 0.0 \
--warmup_steps 20 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--deepspeed "ds_configs/stage2.json" \
--tf32 True \
--report_to none
```
When running the code above, please modify the output_dir and data_path parameters on a case-by-case basis.

## Evaluate Model
```
python eval_next_poi.py --output_dir="../dataset/Gowalla/SanFrancisco-LosAngeles/models/base_models" --city_A=LosAngeles --city_B=SanFrancisco
```
When running the evaluation code above, please modify the parameters on a case-by-case basis.

## Supplementary Evaluation Figures
* Under /figures, we provide supplementary evaluation metrics referenced in our rebuttal.

