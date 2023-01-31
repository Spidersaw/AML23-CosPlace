from datasets.test_dataset import TestDataset
import test
from model import network
import torch
from datetime import datetime
import parser

def load_model(model_path,args):
    model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)
    model_state_dict = torch.load(model_path)
    
    model.load_state_dict(delete_discriminator_layer(model_state_dict))
    return model

def greedy_soup(models_list, val_folder, args):
    sorted_models = []
    val_ds = TestDataset(val_folder,positive_dist_threshold=args.positive_dist_threshold, queries_folder="queries_v1")
    
    for (model,val) in models_list:
    
        model = model.to(args.device)
        model = model.eval()
       
        sorted_models.append((model, val))
        recalls, recalls_str = test.test(args, val_ds, model)
        print(recalls)
        #continue

    sorted_models.sort(key=compare, reverse=True)
    greedy_soup_ingredients = [sorted_models[0][0]]
    greedy_soup_params = sorted_models[0][0].state_dict()
    num_ingredients = len(greedy_soup_ingredients)
    best_val_rec = sorted_models[0][1]

    for i in range(1,len(sorted_models)):
        new_ingredient_params = sorted_models[i][0].state_dict()
        potential_greedy_soup_params = {
                k : greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) + 
                    new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
                for k in new_ingredient_params
            }
        new_model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)

        new_model.load_state_dict(potential_greedy_soup_params)
        new_model = new_model.to(args.device)
        new_model = new_model.eval()
        new_recall, _ = test.test(args, val_ds, new_model)
        print(f'Potential greedy soup val acc {new_recall[0]}, best so far {best_val_rec}.')
        if new_recall[0] > best_val_rec:
            greedy_soup_ingredients.append(sorted_models[i][0])
            best_val_rec = new_recall[0]
            greedy_soup_params = potential_greedy_soup_params
            print(f'Adding to soup.')
    experiment_name= "sample_soup"
    torch.save(greedy_soup_params, f"{experiment_name}.pth")


def compare(m1):
    return m1[1]

def delete_discriminator_layer(model):
    if "discriminator.1.weight" in model:
            del model["discriminator.1.weight"]
            del model["discriminator.1.bias"]
            del model["discriminator.3.weight"]
            del model["discriminator.3.bias"]
            del model["discriminator.5.weight"]
            del model["discriminator.5.bias"]
    return model

if __name__ == "__main__":

    args = parser.parse_arguments(is_training=False)

    base_path = "model/{}"
    models_directories=["cosface/cosface_best_model.pth","sphereface/sphereface_best_model.pth", "arcface/arcface_best_model.pth"]
    val_rec = [52.2,49.7,49.7]
    models_list = []
    for idx, model_path in enumerate(models_directories):
        m = load_model(base_path.format(model_path),args)
        models_list.append((m, val_rec[idx]))
    greedy_soup(models_list, "/content/small/test/", args)
