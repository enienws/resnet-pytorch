import pickle
import os

#Problems:
#NX8yG76fXaw/

input_folder = "/opt/data_in"
output_folder = "/opt/data_out"

if __name__ == "__main__":
    for root, _, files in os.walk(input_folder):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension != ".pickle":
                continue
            if os.path.exists(os.path.join(output_folder, file)) == True:
                print("existing file: ", os.path.join(output_folder, file))
                continue
            print("Processing file: {}".format(file))
            with open(os.path.join(root, file), "rb") as f:
                obj = pickle.load(f)
                obj2 = {}
                # data_obj = {'features_L': features_L.squeeze(0),
                #             'clusters': cluster_ids_x,
                #             'centers': cluster_centers,
                #             'number_of_objects': S,
                #             'class_name': sample_name,
                #             'timeofday': now.strftime("%d/%m/%y %H:%M")}
                obj2['clusters'] = obj['clusters']
                obj2['centers'] = obj['centers']
                obj2['number_of_objects'] = obj['number_of_objects']
                obj2['class_name'] = obj['class_name']
                obj2['timeofday'] = obj['timeofday']

                with open(os.path.join(output_folder, file), "wb") as fwrite:
                    pickle.dump(obj2, fwrite)