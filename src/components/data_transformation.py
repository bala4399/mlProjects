# import sys
# from dataclasses import dataclass

# import numpy as np 
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder

# from src.exception import CustomException
# from src.logger import logging
# import os

# from src.utils import save_object

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config=DataTransformationConfig()

#     def get_data_transformer_object(self):
#         '''
#         This function si responsible for data trnasformation
        
#         '''
#         try:
#             numerical_columns = ["writing_score", "reading_score"]
#             categorical_columns = [
#                 "gender",
#                 "race_ethnicity",
#                 "parental_level_of_education",
#                 "lunch",
#                 "test_preparation_course",
#             ]

#             num_pipeline= Pipeline(
#                 steps=[
#                 ("imputer",SimpleImputer(strategy="median")),
#                 ("scaler",StandardScaler())

#                 ]
#             )
#             print(type(num_pipeline))

#             cat_pipeline=Pipeline(

#                 steps=[
#                 ("imputer",SimpleImputer(strategy="most_frequent")),
#                 ("one_hot_encoder",OneHotEncoder()),
#                 ("scaler",StandardScaler(with_mean=False))
#                 ]

#             )

#             logging.info(f"Categorical columns: {categorical_columns}")
#             logging.info(f"Numerical columns: {numerical_columns}")

#             preprocessor=ColumnTransformer(
#                 [
#                 ("num_pipeline",num_pipeline,numerical_columns),
#                 ("cat_pipelines",cat_pipeline,categorical_columns)

#                 ]


#             )
#             print(type(preprocessor),'******')

#             return preprocessor
        
#         except Exception as e:
#             raise CustomException(e,sys)
        
#     def initiate_data_transformation(self,train_path,test_path):

#         try:
#             train_df=pd.read_csv(train_path)
#             test_df=pd.read_csv(test_path)

#             logging.info("Read train and test data completed")

#             logging.info("Obtaining preprocessing object")

#             # preprocessing_obj=self.get_data_transformer_object()
            
#             train_data = self.dataprepro(train_df)
#             test_data = self.dataprepro(test_df)
            
            

            
#             # numerical_columns = ["writing_score", "reading_score"]

#             # input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
#             # target_feature_train_df=train_df[target_column_name]

#             # input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
#             # target_feature_test_df=test_df[target_column_name]

#             # logging.info(
#             #     f"Applying preprocessing object on training dataframe and testing dataframe."
#             # )

#             # input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
#             # input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

#             # train_arr = np.c_[
#             #     input_feature_train_arr, np.array(target_feature_train_df)
#             # ]
#             # test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

#             # logging.info(f"Saved preprocessing object.")

#             # save_object(

#             #     file_path=self.data_transformation_config.preprocessor_obj_file_path,
#             #     obj=preprocessing_obj

#             # )

#             return (
#                 train_data,
#                 test_data
                
#             )
#         except Exception as e:
#             raise CustomException(e,sys)

#     def dataprepro(self,data):
#         target_column_name="math_score"
#         print(data.columns)
#         target = data[target_column_name]
#         data=data.drop(columns=[target_column_name],axis=1)
        
#         num_features = data.select_dtypes(include=['int64', 'float64'])
#         scaler = StandardScaler()
#         norm_df = scaler.fit_transform(num_features)
#         num_features = pd.DataFrame(norm_df, columns=num_features.columns)
#         imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
#         num_t = imp_mean.fit_transform(num_features)
#         num = pd.DataFrame(num_t, columns=num_features.columns)
#         cat_features = data.select_dtypes('object')
#         enc = LabelEncoder()
#         imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
#         cat_fillval = imp_mode.fit_transform(cat_features)
#         cat_features = pd.DataFrame(cat_fillval, columns=cat_features.columns)
        
#         # cat_features_d = pd.DataFrame(enc.fit_transform(cat_features).toarray())
#         # cat_features_d = pd.get_dummies(cat_features)
#         cat_features_d =enc.fit_transform(cat_features)
            
#         cat_fea = pd.DataFrame(cat_features_d, columns=cat_features_d.columns)
#         norm_catdf = scaler.fit_transform(cat_fea)
#         cat_features = pd.DataFrame(norm_catdf, columns=cat_fea.columns)
#             # preprocessor=load_object(file_path=preprocessor_path)
#             # preprocessor = self.new_method(features)
#         preprocessor = pd.concat([num.reset_index(drop=True), cat_features.reset_index(drop=True),target], axis=1)
        
#         return preprocessor 
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
