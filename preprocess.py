import os
import pandas as pd
import config as cfg

OLD_DATA_PATH = '/home/usrs/shuhei.imai.q4/shuhei.imai/lecture/PBL_ALPS_ALPINE/data'



def file_edit(input_csv_path, output_txt_path):
    csv = pd.read_csv(input_csv_path, low_memory = False)
    with open(output_txt_path, 'a') as f:
        for file_path, label in zip(csv['image'], csv['label']):
            flag_baby, filename = file_path.split('/')[-2], os.path.basename(file_path)
            baby_dir = 'w_baby' if int(flag_baby) else 'wo_baby'
            f.write(os.path.join(baby_dir, filename) + '\n')



def main():
    os.makedirs(os.path.dirname(cfg.path.train_filelist), exist_ok = True)

    file_edit(os.path.join(OLD_DATA_PATH, 'train.csv'), cfg.path.train_filelist)
    file_edit(os.path.join(OLD_DATA_PATH, 'valid.csv'), cfg.path.valid_filelist)
    file_edit(os.path.join(OLD_DATA_PATH, 'test.csv'),  cfg.path.test_filelist)



if __name__ == '__main__':
    main()
