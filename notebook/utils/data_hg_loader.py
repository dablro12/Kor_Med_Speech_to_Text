from typing import Tuple, Union

from datasets import Dataset, Audio, concatenate_datasets
import pandas as pd
import dask.dataframe as dd
# ============================================================================
# Hugging Face Dataset 구조로 변환 (메모리 효율적)
# ============================================================================

def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hugging Face Dataset 변환 전 공통 전처리.
    """
    df_copy = df.copy()
    
    if 'abs_path' not in df_copy.columns:
        raise ValueError("DataFrame에 'abs_path' 컬럼이 없습니다.")

    df_copy['audio'] = df_copy['abs_path'].apply(lambda x: {'path': x})
    
    if 'transcription' in df_copy.columns:
        df_copy['sentence'] = df_copy['transcription']
    
    rename_dict = {}
    if '데이터_카테고리' in df_copy.columns:
        rename_dict['데이터_카테고리'] = 'category'
    if '성별' in df_copy.columns:
        rename_dict['성별'] = 'gender'
    if '연령' in df_copy.columns:
        rename_dict['연령'] = 'age'
    if '지역' in df_copy.columns:
        rename_dict['지역'] = 'region'
    if 'composite_label' in df_copy.columns:
        rename_dict['composite_label'] = 'composite_label'
    
    if rename_dict:
        df_copy = df_copy.rename(columns=rename_dict)
    
    columns_to_keep = ['audio', 'sentence']
    optional_columns = ['category', 'gender', 'age', 'region', 'composite_label']
    for col in optional_columns:
        if col in df_copy.columns:
            columns_to_keep.append(col)
    
    return df_copy[columns_to_keep]
    

def _df_to_dataset(df: pd.DataFrame, sampling_rate: int) -> Dataset:
    dataset = Dataset.from_pandas(df, preserve_index=False)
    
    try:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    except Exception as e:
        print(f"  ⚠ Audio feature cast 실패: {e}")
        print("  path만 저장된 상태로 진행합니다. (audio 컬럼에는 path 딕셔너리만 포함)")
    
    return dataset


# 메모리 효율적인 방법: from_pandas + Dask 분할 처리
def create_hf_dataset_from_df(df: pd.DataFrame, sampling_rate: int = 16000) -> Dataset:
    """
    DataFrame을 Hugging Face Dataset 형식으로 변환 (메모리 효율적)
    """
    print("메모리 효율적인 방식으로 Dataset 생성 중 (pandas)...")
    prepared_df = _prepare_dataframe(df)
    print("  Dataset.from_pandas() 호출 중... (시간이 걸릴 수 있습니다)")
    return _df_to_dataset(prepared_df, sampling_rate)


def create_hf_dataset_from_ddf(ddf: dd.DataFrame, sampling_rate: int = 16000) -> Dataset:
    """
    Dask DataFrame을 Hugging Face Dataset으로 변환 (대용량 지원)
    """
    print("메모리 효율적인 방식으로 Dataset 생성 중 (dask partitions)...")
    delayed_partitions = ddf.to_delayed()
    datasets = []

    for idx, delayed_partition in enumerate(delayed_partitions):
        partition_df = delayed_partition.compute()
        if partition_df.empty:
            continue

        print(f"  Partition {idx + 1}/{len(delayed_partitions)} 변환 중...")
        prepared_df = _prepare_dataframe(partition_df)
        datasets.append(_df_to_dataset(prepared_df, sampling_rate))

    if not datasets:
        raise ValueError("변환 가능한 데이터가 없습니다.")

    if len(datasets) == 1:
        return datasets[0]

    print("  Partition 결과 병합 중...")
    return concatenate_datasets(datasets)

def load_kru_data(
    train_df_path: str,
    test_df_path: str,
    sampling_rate: int = 16000,
    use_dask: bool = True,
    blocksize: Union[str, int, None] = "128MB",
) -> Tuple[Dataset, Dataset]:
    """
    KRU 데이터를 Hugging Face Dataset으로 로드.

    Args:
        train_df_path: 학습 CSV 경로
        test_df_path: 테스트 CSV 경로
        sampling_rate: Audio feature sampling rate
        use_dask: True일 경우 Dask 기반 파이프라인 사용
        blocksize: Dask partition 크기 (bytes, int or str)
    """
    if use_dask:
        print("Dask 기반으로 CSV 로드 중...")
        train_ddf = dd.read_csv(train_df_path, blocksize=blocksize)
        test_ddf = dd.read_csv(test_df_path, blocksize=blocksize)
        train_dataset = create_hf_dataset_from_ddf(train_ddf, sampling_rate=sampling_rate)
        test_dataset = create_hf_dataset_from_ddf(test_ddf, sampling_rate=sampling_rate)
    else:
        print("Pandas 기반으로 CSV 로드 중...")
    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)
    train_dataset = create_hf_dataset_from_df(train_df, sampling_rate=sampling_rate)
    test_dataset = create_hf_dataset_from_df(test_df, sampling_rate=sampling_rate)
    
    return train_dataset, test_dataset


