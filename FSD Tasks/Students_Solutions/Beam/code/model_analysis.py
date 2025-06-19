# import libraries
import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#import data
#df = pd.read_pickle('train.pickle')
#all_train_data = np.stack(df.sensor_data.to_numpy())
#all_train_labels = df.label.to_numpy()

df_test = pd.read_pickle('test.pickle')
test_data = np.stack(df_test.sensor_data.to_numpy())
test_labels = df_test.label.to_numpy()

#define class names for confusion matrices
class_names = ["0","1","2","3","4","5","6","7","8","9","10","11"]

def save_confusion_matrix(model, model_name):
    pred_labels = np.argmax(model.predict(test_data), axis=1)
    cm = confusion_matrix(test_labels, pred_labels)

    model_name_short = model_name.removeprefix("models/")

    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("figures/" + model_name_short + ".png") 

all_cnn_names = ["models/Model(Conv+2xDense)-id_df7287b351af4491b6c0f53f1f02d1fb",
                "models/Model(2xConv+2xDense)-id_e0e5bc5e81ab4fa2a79f3c30a9ea64f4",
                "models/Model(2xConv+4xDense)-id_33f91222f49545f092360b24e0ce9b6b",
                "models/Model(2xConv+4xDense)-id_77d30cda7ef14787b7da256b28978173",
                "models/Model(3xConv+4xDense)-id_7bd97e8acae6480097aee2743b2aafe4",
                "models/Model(2xConv+3xDense)-id_bfcd3a978c784d7cb16596050349048e",	
                "models/Model(3xConv+3xDense)-id_e98a318509074a28b0496d7f4414f7cc",	
                "models/Model(3xConv+2xDense)-id_4f92b2cb309c4827b4a0fc75dd73a2ab",	
                "models/Model(2xConv+4xDense)-id_8011af44900240149b6dfa1789a9f79d",
                "models/Model(2xConv+3xDense)-id_503c2e71e73f4bdba6ea6598a00fc3fd",	
                "models/Model(2xConv+2xDense)-id_5920445026534abfacf28ca06ea17a80",
                "models/Model(2xConv+4xDense)-id_2aab541d188947f78534aebf69dfcfae",
                "models/Model(2xConv+4xDense)-id_a95669dae18348b2af9bad98ca88ff0c",
                "models/Model(2xConv+2xDense)-id_f9b2eb833730437ba627b36af58bee3c",	
                "models/Model(2xConv+3xDense)-id_62f08c82dd124436a022bbe54f61f9d0",	
                "models/Model(Conv+3xDense)-id_3a3a5279523342d2a06e10096af9e22f",
                "models/Model(Conv+3xDense)-id_e23c63e085eb42b7924e9420c26a76a7",	
                "models/Model(3xConv+3xDense)-id_ee0495ad1de34da2837694e712c67797",	
                "models/Model(2xConv+4xDense)-id_3b36cfa42bcf436ba62a1a2693143154",
                "models/Model(Conv+3xDense)-id_3353ddd9f8324b029c6e127a146d83ed"]

all_fcn_names=["models/Model(4xDense)-id_b7847dc1c24b4f61affc25c2d62cc4e7",
                "models/Model(5xDense)-id_ef744be16a5f4ac194165cc48d48deaa",
                "models/Model(3xDense)-id_46e1227238b0459b8c97162d7ef4c40b",
                "models/Model(7xDense)-id_fef2faf3c8bb4f6292b8a3ac546d9059",
                "models/Model(2xDense)-id_531efe92d5da46eaa3f7f7a9cfb2bdc1",
                "models/Model(4xDense)-id_f9440973bc67471682dc20b12cb92cd4",	
                "models/Model(6xDense)-id_857fbe79c027484da0455874e5bfe251",
                "models/Model(4xDense)-id_780dbf8d1700467fb725b36a9b7d241b",
                "models/Model(4xDense)-id_b6073991e83746b1a8be0bc91bf21341",
                "models/Model(7xDense)-id_234ff050e7144a82b0c42f41399e518e",	
                "models/Model(5xDense)-id_24c3fc95f87847d889229ad2dd29db0a",
                "models/Model(3xDense)-id_bcd144d71b00405ab46d536d78f2c19c",
                "models/Model(5xDense)-id_1e537a694aaa49a99142cd41254abfc8",
                "models/Model(2xDense)-id_bb0023a5b755454caccaba4ff8dbbe5a",
                "models/Model(4xDense)-id_26625201e2734eaca85ae43bd9ddb8ff",	
                "models/Model(6xDense)-id_115f178a86e14b9b83340b2d741d9a85",	
                "models/Model(3xDense)-id_f4a3a4a36b1240059ac4bd422dfeb42d",	
                "models/Model(3xDense)-id_9c2b95a785db45e8943db93df11561c9",
                "models/Model(6xDense)-id_8361115f74054ea683e97deb9f7b4696",
                "models/Model(5xDense)-id_a48d5bbb4abc4469b592a22acbdacaf1"]

for model_name in all_cnn_names:
    model = tf.keras.models.load_model(model_name + ".keras")
    save_confusion_matrix(model, model_name = model_name)