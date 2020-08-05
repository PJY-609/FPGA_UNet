from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd


IMAGE_DIR = '/media/_data/datasets/CVC-ClinicDB/Original'
MASK_DIR = '/media/_data/datasets/CVC-ClinicDB/Ground Truth'
VAL_RATIO = 0.1
DOC_SAVE_DIR = '/home/juezhao/FPGA_UNet/'
RANDOM_STATE = 13


def main():
	image_fps = [str(fp) for fp in Path(IMAGE_DIR).iterdir()]
	mask_fps = [str(fp) for fp in Path(MASK_DIR).iterdir()]

	train_imgs, val_imgs, train_msks, val_msks = train_test_split(image_fps, mask_fps, test_size=VAL_RATIO, random_state=RANDOM_STATE)

	train_df = pd.DataFrame({'img':train_imgs, 'msk':train_msks})
	train_fp = Path(DOC_SAVE_DIR) / 'train.csv'
	train_df.to_csv(str(train_fp))

	val_df = pd.DataFrame({'img':val_imgs, 'msk':val_msks})
	val_fp = Path(DOC_SAVE_DIR) / 'val.csv'
	val_df.to_csv(str(val_fp))


if __name__ == '__main__':
	main()
