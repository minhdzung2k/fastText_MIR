from Embedding_Preprocessing.encoder_tuple_level import TupleTokenizationMode
from fasttext_back_end import FastTextBackEnd

def main():
    map_file_path = "./Embedding_Preprocessing/opt_encoder.tsv" 
    config_file_path = "./Configuration/config/config_2"
    dataset_file_path = "./NTCIR-12_MathIR_Wikipedia_Corpus/MathTagArticles"
    queries_directory_path = "./TestQueries"
    model_file_path = "opt_model"
    embedding_type = TupleTokenizationMode(3) # Both_Separated - Type and Value
    tokenize_all = False # Tokenize all elements (such as numbers and text)
    tokenize_number = True # Tokenize the numbers or not
    is_wiki = True # The dataset is wiki or not
    read_slt = False # True: SLT, False: OPT
    do_retrieval = True # True to do the retrieval on NTCIR-12 dataset 
    ignore_full_relative_path = True # Ignore full relative path
    train_model = True  # True for training a new model and False for loading a model
    system = FastTextBackEnd(config_file=config_file_path, path_data_set=dataset_file_path, is_wiki=is_wiki,
                               read_slt=read_slt, queries_directory_path=queries_directory_path)

    if train_model:
        dictionary_formula_tuples_collection = system.train_model(
            map_file_path=map_file_path,
            model_file_path=model_file_path,
            embedding_type=embedding_type, ignore_full_relative_path=ignore_full_relative_path,
            tokenize_all=tokenize_all,
            tokenize_number=tokenize_number
        )
        if do_retrieval:
            retrieval_result = system.retrieval(dictionary_formula_tuples_collection,
                                                embedding_type, ignore_full_relative_path, tokenize_all,
                                                tokenize_number
                                                )
            system.create_result_file(retrieval_result, "./Retrieval_Results/opt_res.tsv", "2")
    else:

        dictionary_formula_tuples_collection = system.load_model(
            map_file_path=map_file_path,
            model_file_path=model_file_path,
            embedding_type=embedding_type, ignore_full_relative_path=ignore_full_relative_path,
            tokenize_all=tokenize_all,
            tokenize_number=tokenize_number
        )
        if do_retrieval:
            retrieval_result = system.retrieval(dictionary_formula_tuples_collection,
                                                embedding_type, ignore_full_relative_path, tokenize_all,
                                                tokenize_number
                                                )
            system.create_result_file(retrieval_result, "./Retrieval_Results/opt_res.tsv", "2")
if __name__ == "__main__":
    main()