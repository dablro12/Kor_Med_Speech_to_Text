from glob import glob 
import pandas as pd
import simdjson as json 
import os

def build_kru_dataframe_and_save(data_root_dir, tag_file_path, save_csv_path):
    # Load tag mapping
    with open(tag_file_path, 'r', encoding='utf-8') as f:
        tag_data = json.loads(f.read())

    # Create mapping from column name to index_order and values
    column_to_index = {}
    column_to_values = {}
    for key, value in tag_data.items():
        column_name = key.strip()  # Remove trailing space if exists
        index_order = value["-index_order"]
        column_to_index[column_name] = index_order
        column_to_values[column_name] = value["values"]

    # Helper function to map code to actual value
    def map_code_to_value(column_name, code):
        if code is None:
            return None
        values_dict = column_to_values.get(column_name, {})
        return values_dict.get(code, code)  # Return code itself if not found in mapping

    # Collect all wav files and parse filenames
    rows = []
    for file_path in glob(f"{data_root_dir}/**/**/**/*.wav"):
        abs_path = os.path.abspath(file_path)
        filename = os.path.basename(file_path)

        # Split filename by "-"
        parts = filename.replace('.wav', '').split('-')

        # Extract values based on index_order and map to actual values
        row = {'abs_path': abs_path}

        # 데이터_항목 (index_order: 2)
        if len(parts) > column_to_index['데이터_항목']:
            code = parts[column_to_index['데이터_항목']]
            row['데이터_항목'] = map_code_to_value('데이터_항목', code)
        else:
            row['데이터_항목'] = None

        # 데이터_카테고리 (index_order: 3)
        if len(parts) > column_to_index['데이터_카테고리']:
            code = parts[column_to_index['데이터_카테고리']]
            row['데이터_카테고리'] = map_code_to_value('데이터_카테고리', code)
        else:
            row['데이터_카테고리'] = None

        # 성별 (index_order: 4)
        if len(parts) > column_to_index['성별']:
            code = parts[column_to_index['성별']]
            row['성별'] = map_code_to_value('성별', code)
        else:
            row['성별'] = None

        # 연령 (index_order: 5)
        if len(parts) > column_to_index['연령']:
            code = parts[column_to_index['연령']]
            row['연령'] = map_code_to_value('연령', code)
        else:
            row['연령'] = None

        # 지역 (index_order: 6)
        if len(parts) > column_to_index['지역']:
            code = parts[column_to_index['지역']]
            row['지역'] = map_code_to_value('지역', code)
        else:
            row['지역'] = None

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows, columns=['abs_path', '데이터_항목', '데이터_카테고리', '성별', '연령', '지역'])

    print(f"Total files processed: {len(df)}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataFrame info:")
    print(df.info())

    # Save to CSV
    df.to_csv(save_csv_path, encoding="utf-8", index=False)
    return df

# Example usage:
# df = build_kru_dataframe_and_save("/workspace/kru_data", "/workspace/kru_data/data_tag.json", "data_source.csv")