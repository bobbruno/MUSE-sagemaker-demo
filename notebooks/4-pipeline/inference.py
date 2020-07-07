
import argparse
import json
import logging
import os
import sys
import boto3
import time
import csv
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client

def invoke_endpoint(row: pd.Series, region, endpoint):
    runtime = boto3.client('runtime.sagemaker', region_name=region)
    payload = row.description
    response = runtime.invoke_endpoint(EndpointName=endpoint,
                                       ContentType='text/csv',
                                       Body=payload)
    result = json.loads(response['Body'].read().decode())['output']
    return(str(result))

def invoke_inference(df, region, endpoint):
    return(df.apply(invoke_endpoint, axis=1, region=region, endpoint=endpoint))

def gen_inference(source_data_dir, inference_data_dir, endpoint_name, region, block_size='32MB', sample=1.0):
    print("---------------------------------------------")
    print(f"Loading data from {source_data_dir}.")
    if sample < 1.0:
        print(f"Taking a fraction of {sample:0.2f} of the data")
    print("---------------------------------------------")
    data = dd.read_csv(
        f'{source_data_dir}/dataset-*.csv', header=0, 
        usecols=['description', 'title'],
        blocksize=block_size,
    ).repartition(partition_size=block_size).sample(frac=sample)
    
    data['embedding'] = data.map_partitions(
        invoke_inference,
        meta=pd.Series(name='embedding', dtype='U'),
        region=region,
        endpoint=endpoint_name
    )
    print(f"Saving inference dataset to {inference_data_dir}")
    data.to_csv(f'{inference_data_dir}/dataset-*.csv', compute=True, index=False, quoting=csv.QUOTE_NONNUMERIC)    


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
    parser.add_argument("--inference-data", type=str, default="inference-dataset")
    parser.add_argument("--endpoint-name", type=str, default="muse-large")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--block-size", type=str, default="32MB")
    parser.add_argument("--scheduler-ip", type=str, default=sys.argv[-1])
    parser.add_argument("--sample", type=float, default=1.0)
    args, _ = parser.parse_known_args()
    
    # Get processor scrip arguments
    args_iter = iter(sys.argv[1:])
    script_args = dict(zip(args_iter, args_iter))
    return(args, script_args)


if __name__ == '__main__':
    inputs, outputs = parse_processing_job_config()
    args, script_args = parse_arguments()
    start_dask_cluster(args.scheduler_ip)
    
    print('----------------------------------------------------')
    print('Starting inference')
    print('----------------------------------------------------')
    gen_inference(
        source_data_dir=inputs[args.data_to_process], 
        inference_data_dir=outputs[args.inference_data],
        endpoint_name=args.endpoint_name,
        region=args.region,
        block_size=args.block_size,
        sample=args.sample
    )
    print('----------------------------------------------------')
    print('Inference finished')
    print('----------------------------------------------------')