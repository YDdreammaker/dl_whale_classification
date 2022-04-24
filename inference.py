import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from data import get_valid_transforms, ImageDataset
from model import HappyWhaleModel
import argparse


def inference_logit(model, state_dict_path, test_loader, tta=False):
    print(state_dict_path)
    pt_load = torch.load(state_dict_path)
    try: # ['model'] for checkpoints
        model.load_state_dict(pt_load['model'])
    except:
        model.load_state_dict(pt_load)
    model.eval()

    temp = []
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)
            
            if tta:
                x = torch.flip(x, dims=[-1])
                
            logit, _ = model(x, y)

            logit = logit.detach().cpu()
            temp.extend(logit)

    res = torch.stack(temp, dim=0).squeeze()
    print(res.shape)
    return res


def inference_embedding(model, state_dict_path, test_loader):
    pt_load = torch.load(state_dict_path)
    model.load_state_dict(pt_load['model'])
    model.eval()

    temp = []
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)

            emb = emb.detach().cpu()
            temp.extend(emb)

    res = torch.stack(temp, dim=0).squeeze()
    return res


def main_embedding(batch_size, model_name, emb_size, model_path, image_size,
        train_root, test_root, train_image_root, test_image_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_test = pd.read_csv(test_root)
    test_image_path = df_test.image.values
    test_target = [0] * len(df_test)
    valid_transform = get_valid_transforms(image_size)

    df_train = pd.read_csv(train_root)
    
    # drop error images
    drop_index = [11604,15881,16782,21966,23306,24862,25895,29468,31831,35805,37176,40834,47480,48455,36710,47161,23626]
    df_train = df_train.drop(drop_index, axis=0).reset_index(drop=True)

    classes = df_train.individual_id.unique()
    inv_class_map = dict(enumerate(classes, 0))
    class_map = {v: k for k, v in inv_class_map.items()}
    df_train.individual_id = df_train.individual_id.map(class_map)
    train_target = df_train.individual_id.values

    train_image_path = df_train.image.values

    test_set = ImageDataset(test_image_path, valid_transform, root=test_image_root, target=test_target)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    train_set = ImageDataset(train_image_path, valid_transform, root=train_image_root, target=train_target)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, pin_memory=True)

    model = HappyWhaleModel(model_name=model_name, emb_size=emb_size)
    model.to(device)

    test_embed = inference_embedding(model, model_path, test_loader)
    train_embed = inference_embedding(model, model_path, train_loader)
    
    return test_embed, train_embed


def main_logit(batch_size, emb_size, model_path, test_root, test_image_root, n_classes=15587):
    print('make output logits')

    df_test = pd.read_csv(test_root)
    test_image_path = df_test.image.values
    test_target = [0] * len(df_test)
    print('len:', len(df_test))
    print(model_path)
    for path in model_path:
        info = path.split('_')
        print(info)
        try:
            image_size = int(info[5])
        except: image_size = int(info[4])
        save_name = '_'.join(info[1:-3])

        if info[3] == 'v2' and info[4] == 'm':
            model_name = 'tf_efficientnetv2_m'
        elif info[3] == 'v2' and info[4] == 'l':
            model_name = 'tf_efficientnetv2_l'
        elif info[3] == 'b5':
            model_name = 'tf_efficientnet_b5_ap'
        elif info[3] == 'b6':
            model_name = 'tf_efficientnet_b6_ap'
        elif info[3] == 'b7':
            model_name = 'tf_efficientnet_b7_ap'
        elif 'conv' in info[3]:
            model_name = 'convnext_base'

        print("path, name, model name, image size", 
              path, save_name, model_name, image_size)
        
        valid_transform = get_valid_transforms(image_size)
        test_set = ImageDataset(test_image_path, valid_transform, root=test_image_root, target=test_target)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=8, pin_memory=True)

        model = HappyWhaleModel(model_name=model_name, emb_size=emb_size)
        model.to(device)

        logit = inference(model, path, test_loader)
        assert logit.shape[0] == len(df_test)
        assert logit.shape[1] == n_classes
        torch.save(logit, f'../{save_name}_output_logit.pt')
        
        tta_logit = inference(model, path, test_loader, tta=True)
        torch.save(tta_logit, f'../{save_name}_tta_output_logit.pt')
    return


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Embedding or Logit inference')
    
    parser.add_argment('--inference_type', '-t', type=str, default='embedding'
                      help='Choose embedding inference or logit inference')
    
    args = parser.parse_args()
    
    if args.inference_type == 'enbedding':

        main_embedding(
            batch_size=64,
            emb_size=2048,
            train_root='../data3/train.csv',
            test_root='../data3/sample_submission.csv',
            test_image_root='../data3/test_detec_512_v5/',
            train_image_root='../data3/train_detec_512_v5/',
            image_size= 384, 384
            model_name='tf_efficientnet_b5_ns', 
            model_path='../saved/checkpoint/tf_efficientnetv2_m/000.pt',
            )
    else:

        main_logit(
            batch_size=64,
            emb_size=2048,
            test_root='../data3/sample_submission.csv',
            test_image_root='../data3/test_detec_512_v5/',
            model_path=glob('../modeltoprocess/*'),
            )
        