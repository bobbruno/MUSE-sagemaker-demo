import argparse
import json
import logging
import os
import sys
import time
import csv
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from langdetect import detect as detect_lang, DetectorFactory
DetectorFactory.seed = 0


SUPPORTED_LANGUAGES = {'ar', 'nl', 'en', 'de', 'fr', 'it', 'pt', 'es', 'ja', 'ko', 'ru', 'pl', 'tr', 'zh', 'zh-tw', 'th'}

def detect_descr_lang(row: pd.Series) :
    if row.lang == 'en':  # Accept that english is correct
        return('en')
    else:
        try:
            detected_lang = detect_lang(row.description)  # If it can't detect the language, it returns 'unknown'
        except:
            detected_lang = 'unknown'
        if (row.lang == 'ru' and detected_lang != 'en'):   # If reported russion and detected not english, assume reported is correct
            detected_lang = 'ru'
        elif(detected_lang in {'zh-cn', 'ko', 'zh-tw'}):   # Consolidate all chinese variants and korean as general chinese.
            detected_lang = 'zh'
        return(detected_lang)
    
def detect_df_lang(df):
    return(df.apply(detect_descr_lang, axis=1))

def report_transformation(report_dest_dir, max_descr_length, langs_orig_dataset, truncated_descriptions):
    # TODO: Send reports to a file - currently sending to logs
    print("Generating reports...")
    print("---------------------------------------------")
    print(f"{langs_orig_dataset.num_books.sum()} records in total in the original dataset.")
    print(f"Number of books per language in the original dataset:\n{langs_orig_dataset.sort_values('num_books', ascending=False).to_string()}")
    print("---------------------------------------------")
    if max_descr_length > 0 and truncated_descriptions is not None:
        num_truncated = truncated_descriptions.shape[0].compute()
        print(f"{num_truncated} descriptions were truncated at {max_descr_length} characters")

def dump_rejected(rejected_dest_dir, dropped_na_description, dropped_non_supported_lang, dataset_langs_filtered_out, english_wrong, short_descriptions):
    def dump_df_one_file(df, dest):
        try:
            next(df.iterrows())  # Checking if there is anything on the dataframe
            df.to_csv(dest, compute=True, index=False, single_file=True, quoting=csv.QUOTE_NONNUMERIC)
            return(len(df))
        except StopIteration:
            return(0)
    print("---------------------------------------------")
    print("Saving rejected records...")
    cum_num_dropped = 0
    num_rows = dump_df_one_file(dropped_na_description, f'{rejected_dest_dir}/dropped_na.csv')
    cum_num_dropped += num_rows
    print(f"{num_rows} records rejected because description is empty.")
    num_rows = dump_df_one_file(dropped_non_supported_lang, f'{rejected_dest_dir}/dropped_non_supported_lang.csv')
    cum_num_dropped += num_rows
    print(f"{num_rows} records rejected because language is not supported.")
    num_rows = dump_df_one_file(dataset_langs_filtered_out, f'{rejected_dest_dir}/lang_filtered_out.csv')
    cum_num_dropped += num_rows
    print(f"{num_rows} records rejected because language was filtered out.")
    num_rows = dump_df_one_file(english_wrong, f'{rejected_dest_dir}/english_wrong.csv')
    cum_num_dropped += num_rows
    print(f"{num_rows} records rejected because english was wrongly reported as language.")
    num_rows = dump_df_one_file(short_descriptions, f'{rejected_dest_dir}/short_descriptions.csv')
    cum_num_dropped += num_rows
    print(f"{num_rows} records rejected because description was too short.")
    print(f"{cum_num_dropped} records rejected in total.")
    print('Rejected records saved...')
    print("---------------------------------------------")


def save_description(dest_file, df):
    print("---------------------------------------------")
    print(f"Saving descriptions to {dest_file}")
    with open(dest_file, 'w') as dest:
        for descr in df.iteritems():
            try:
                dest.write(f'{{"description": {json.dumps(descr[1])}}}\n')
            except e:
                print(f'Description rejected: {descr}')
    print("Descriptions saved.")
    print("---------------------------------------------")


def gen_cleaned_data(source_data_dir, dest_data_dir, descr_data_dir, rejected_data_dir, reports_dir, drop_languages, max_descr_length,
                     supported_languages=SUPPORTED_LANGUAGES, block_size='32MB', sample=1.0) -> dd.DataFrame: 
    print("---------------------------------------------")
    print(f"Loading data from {source_data_dir}.")
    if sample < 1.0:
        print(f"Taking a fraction of {sample:0.2f} of the data")
    print(f"Rejected data will be sent to {rejected_data_dir}.")
    print(f"Reports (if any) will be sent to {reports_dir}.")
    print("---------------------------------------------")
    raw_df = dd.read_csv(
        f'{source_data_dir}/dataset.csv', header=0, 
        usecols=['description', 'authors', 'categories', 'lang', 'title'],
        blocksize=block_size,
    ).repartition(partition_size=block_size).sample(frac=sample)
    
    langs_orig_df = raw_df[['lang', 'title']].groupby('lang').count().compute().rename(columns={'title': 'num_books'})
    
    dropped_na_description_df = raw_df[raw_df.description.isna()]
    non_na_df = raw_df[~ raw_df.description.isna()]
    
    # Truncating descriptions if requested
    if max_descr_length > 0:
        truncated_descriptions_df = non_na_df[non_na_df.description.str.len() > max_descr_length]
        non_na_df.description = non_na_df.description.str.slice(stop=max_descr_length)
    else:
        truncated_descriptions_df = None
    non_na_df['descr_len_words'] = non_na_df.map_partitions(lambda df: df.description.apply(lambda t: len(t.split(' '))), meta=pd.Series(name='descr_len_words', dtype='i4'))
    non_na_df['detected_lang'] = non_na_df.map_partitions(detect_df_lang, meta=pd.Series(name='detected_lang', dtype='U'))
    
    dropped_non_supported_lang_df = non_na_df[~(non_na_df.lang.isin(supported_languages) | non_na_df.detected_lang.isin(supported_languages))]
    supported_lang_df = non_na_df[non_na_df.lang.isin(supported_languages) | non_na_df.detected_lang.isin(supported_languages)]
    
    langs_filtered_out_df = supported_lang_df[supported_lang_df.lang.isin(drop_languages)]
    filtered_df = supported_lang_df[~supported_lang_df.lang.isin(drop_languages)]  # Removing languages we were asked to filter out
    
    # Keep detected non-english (for language diversity) or detected and reported english (drop all reported english but detected something else)
    english_wrong_df = filtered_df[(filtered_df.detected_lang == 'en') & ~(filtered_df.detected_lang == filtered_df.lang)]
    non_english_or_lang_match_df = filtered_df[(filtered_df.detected_lang != 'en') | (filtered_df.detected_lang == filtered_df.lang)]

    # Removing very short descriptions from dataset. We keep all chinese because the language is more expressive.
    short_descriptions_df = non_english_or_lang_match_df[(non_english_or_lang_match_df.descr_len_words < 8) &
                                                (non_english_or_lang_match_df.detected_lang != 'zh')]
    processed_df = non_english_or_lang_match_df[(non_english_or_lang_match_df.descr_len_words >= 8) |
                                                (non_english_or_lang_match_df.detected_lang == 'zh')]  
    
    report_transformation(reports_dir, max_descr_length, langs_orig_df, truncated_descriptions_df)
    print(f"Saving transformed dataset to {dest_data_dir}")
    processed_df.to_csv(f'{dest_data_dir}/dataset-*.csv', compute=True, index=False, quoting=csv.QUOTE_NONNUMERIC)
    save_description(f'{descr_data_dir}/dataset.jsonl', processed_df.description)
    dump_rejected(rejected_data_dir, dropped_na_description_df, dropped_non_supported_lang_df, langs_filtered_out_df, english_wrong_df, short_descriptions_df)
    
    
def start_dask_cluster(scheduler_ip):
    # Start the Dask cluster client
    try:
        client = Client("tcp://{ip}:8786".format(ip=scheduler_ip))
        logging.info("Cluster information: {}".format(client))
    except Exception as err:
        logging.exception(err)


def parse_processing_job_config(config_file="/opt/ml/config/processingjobconfig.json"):
    with open(config_file, "r") as config_file:
        config = json.load(config_file)
    inputs = {in_path["InputName"]: in_path["S3Input"]["LocalPath"] for in_path in config["ProcessingInputs"]}
    outputs = {out_path["OutputName"]: out_path["S3Output"]["LocalPath"] for out_path in config["ProcessingOutputConfig"]["Outputs"]}
    return (inputs, outputs)
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-to-process", type=str, default="dataset")
    parser.add_argument("--data-to-generate", type=str, default="processed-dataset")
    parser.add_argument("--descriptions", type=str, default="descriptions-dataset")
    parser.add_argument("--rejected-data", type=str, default="rejected-dataset")
    parser.add_argument("--reports", type=str, default="dataset-reports")
    parser.add_argument("--supported-languages", nargs='+', default=SUPPORTED_LANGUAGES)
    parser.add_argument("--max-description-length", type=int, default=1024)
    parser.add_argument("--drop-languages", nargs="+", default=['ja', 'ar', 'ko', 'th'])
    parser.add_argument("--block-size", type=str, default="32MB")
    parser.add_argument("--scheduler-ip", type=str, default=sys.argv[-1])
    parser.add_argument("--sample", type=float, default=1.0)
    args, _ = parser.parse_known_args()
    
    print(f'Supported Languages: {args.supported_languages}')
    print(f'Languages {args.drop_languages} will be dropped from the dataset')
    # Get processor scrip arguments
    args_iter = iter(sys.argv[1:])
    script_args = dict(zip(args_iter, args_iter))
    return(args, script_args)

if __name__ == '__main__':
    inputs, outputs = parse_processing_job_config()
    args, script_args = parse_arguments()
    start_dask_cluster(args.scheduler_ip)
    
    print('----------------------------------------------------')
    print('Starting processing')
    print('----------------------------------------------------')
    gen_cleaned_data(
        source_data_dir=inputs[args.data_to_process], 
        dest_data_dir=outputs[args.data_to_generate],
        descr_data_dir=outputs[args.descriptions],
        rejected_data_dir=outputs[args.rejected_data],
        reports_dir=outputs[args.reports],
        drop_languages=set(args.drop_languages), 
        max_descr_length=args.max_description_length,
        supported_languages=args.supported_languages,
        block_size=args.block_size,
        sample=args.sample
    )
    print('----------------------------------------------------')
    print('Processing finished')
    print('----------------------------------------------------')