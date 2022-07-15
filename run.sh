#train UAS-parsers
python3 train_UAS.py --train_option in_sentence --dest_path in_sentence_temp.pt
python3 train_UAS.py --train_option between_sentence --dest_path between_sentence_temp.pt --epochs 3

#train in-sentence relations
python3 train_relations.py --train_option in_sentence_bert --epochs 3 --path_dev_data preprocessed_data/sci_dev.data --dest_path in_bert_temp.pt
python3 train_relations.py --train_option in_sentence_lstm  --epochs 30 --path_dev_data preprocessed_data/sci_dev.data  --path_bert in_bert_temp.pt --learning_rate 4e-5 --dest_path in_tagger_temp.pt

#train between-sentence relations
python3 train_relations.py --train_option between_sentence_bert --dest_path between_bert_temp.pt --epochs 3 --path_dev_data preprocessed_data/sci_dev.data
python3 train_relations.py --train_option between_sentence_lstm  --epochs 20 --path_dev_data preprocessed_data/sci_dev.data  --path_bert between_bert_temp.pt  --dest_path between_tagger_temp.pt --learning_rate 2e-3
#evaluate trained_models
python3 evaluate.py --path_in_sentence_model in_sentence_temp.pt --path_between_sentence_model between_sentence_temp.pt --path_relation_bert in_bert_temp.pt --path_lstm_tagger in_tagger_temp.pt --path_between_bert between_bert_temp.pt --path_between_tagger between_tagger_temp.pt
