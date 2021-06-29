"""
read data from datasource and safe it to
data raw for further preprocess
"""
import os
from get_data import read_params, get_data
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import librosa
from tqdm import tqdm


def feature_extractor(file,config):
    audio,sample_rate=librosa.load(file,res_type="kaiser_fast")
    mfccs_features=librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=config["estimators"]["n_mfcc"])
    mfccs_features=np.mean(mfccs_features.T,axis=0)
    return mfccs_features


def load_and_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    audio_data_path=config["data_source"]["s2_source"]
    extracted_features=[]
    for index_num,row in tqdm(df.iterrows()):
        file_name=os.path.join(os.path.abspath(audio_data_path),'fold'+str(row['fold'])+'//',str(row['slice_file_name']))
        final_class_label=row["class"]
        data_ex=feature_extractor(file_name,config)
        extracted_features.append([data_ex,final_class_label])
    extracted_features_df=pd.DataFrame(extracted_features,columns=["features","class"])
    
    encoder = LabelEncoder()
    extracted_features_df["class"]=encoder.fit_transform(extracted_features_df["class"])
    print(extracted_features_df.head(10))
    extracted_features_df.to_csv(raw_data_path, index=False)


 # run comment
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)