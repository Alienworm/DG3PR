import argparse

from utils.logger import Logger
from utils.file_util import load_csv_file, save_to_dill


def parse_args():
    parser = argparse.ArgumentParser(description='Data Preprocess')
    parser.add_argument(
        '--shipping_data_path',
        type=str,
        default='raw_data/ADS_AI_MRT_SALES_SHIPPING.csv',
        help='shipping data path'
    )
    parser.add_argument(
        '--breed_class_data_path',
        type=str,
        default='raw_data/DIM_BREEDS_FILECLASS.csv',
        help='breed class data path'
    )
    parser.add_argument(
        '--cust_type_data_path',
        type=str,
        default='raw_data/DIM_CUST.csv',
        help='cust type data path'
    )
    parser.add_argument(
        '--processed_shipping_data_path',
        type=str,
        default='data/processed_shipping_data.dill',
        help='processed shipping data path'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger = Logger()

    logger.divider('Data Preprocess Start')
    shipping_data = load_csv_file(args.shipping_data_path)
    breed_class_data = load_csv_file(args.breed_class_data_path)
    cust_type_data = load_csv_file(args.cust_type_data_path)
    logger.info('Raw Data Load Done!')

    shipping_data = shipping_data[['shipping_dt', 'org_inv_dk', 'forsale_breeds_nm', 'cust_dk', 'province_nm', 'forsale_breeds_dk', 'fg_grade_nm', 'sale_qty', 'sale_price', 'avg_wt', 'guide_price']]

    # drop null
    shipping_data_shape = shipping_data.shape
    not_null_columns = ['shipping_dt', 'org_inv_dk', 'forsale_breeds_nm', 'cust_dk', 'province_nm', 'forsale_breeds_dk', 'fg_grade_nm', 'sale_qty', 'sale_price', 'avg_wt', 'guide_price']
    shipping_data = shipping_data.dropna(subset=not_null_columns)
    logger.info('Before Drop Null: {}, After Drop Null: {}'.format(shipping_data_shape, shipping_data.shape))

    # if sale_qty < 0 or sale_price < 0 or avg_wt < 0 or guide_price < 0, drop
    shipping_data_shape = shipping_data.shape
    not_negative_columns = ['sale_qty', 'sale_price', 'avg_wt', 'guide_price']
    shipping_data = shipping_data[(shipping_data[not_negative_columns] >= 0).all(axis=1)]
    logger.info('Before Drop Negative: {}, After Drop Negative: {}'.format(shipping_data_shape, shipping_data.shape))

    # drop duplicates
    shipping_data_shape = shipping_data.shape
    shipping_data = shipping_data.drop_duplicates()
    logger.info('Before Drop Duplicates: {}, After Drop Duplicates: {}'.format(shipping_data_shape, shipping_data.shape))

    shipping_data_shape = shipping_data.shape
    shipping_data = shipping_data.merge(
        breed_class_data[['org_inv_dk', 'forsale_breeds_dk', 'breeds_class_nm']],
        how='left',
        on=['org_inv_dk', 'forsale_breeds_dk']
    )
    shipping_data = shipping_data.dropna(subset=['breeds_class_nm'])
    logger.info('Before Merge Breed Class: {}, After Merge Breed Class: {}'.format(shipping_data_shape, shipping_data.shape))

    shipping_data_shape = shipping_data.shape
    shipping_data = shipping_data.merge(
        cust_type_data[['cust_dk', 'inner_company_ind']],
        how='left',
        on=['cust_dk']
    )
    shipping_data = shipping_data.dropna(subset=['inner_company_ind'])
    logger.info('Before Merge Cust Type: {}, After Merge Cust Type: {}'.format(shipping_data_shape, shipping_data.shape))

    save_to_dill(shipping_data, args.processed_shipping_data_path)
    logger.divider('Data Preprocess Done', end=True)


if __name__ == '__main__':
    main()